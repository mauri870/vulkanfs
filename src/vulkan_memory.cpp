#include "vulkan_memory.hpp"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>

namespace {
    // Non-temporal store loop for Write-Combining (BAR/SAM) destinations.
    // Plain memcpy uses regular stores which stall on WC buffer flushes;
    // movnt + sfence lets the CPU pipeline PCIe writes efficiently.
    //
    // _mm256_stream_si256 requires 32-byte-aligned destination. Writes to
    // arbitrary file offsets may land at unaligned positions within the mapped
    // buffer, so we copy an unaligned prefix with plain memcpy first.
    inline void memcpy_nt(void* __restrict__ dst, const void* __restrict__ src, size_t n) {
        // Align dst to 32 bytes before starting NT stores.
        size_t prefix = (32 - ((uintptr_t)dst & 31)) & 31;
        if (prefix > n) prefix = n;
        if (prefix) {
            std::memcpy(dst, src, prefix);
            dst = static_cast<char*>(dst) + prefix;
            src = static_cast<const char*>(src) + prefix;
            n  -= prefix;
        }

        auto d = reinterpret_cast<__m256i*>(dst);
        auto s = reinterpret_cast<const __m256i*>(src);
        const size_t chunks = n / sizeof(__m256i);
        for (size_t i = 0; i < chunks; i++)
            _mm256_stream_si256(d + i, _mm256_loadu_si256(s + i));
        _mm_sfence();
        // handle trailing bytes (n not a multiple of 32)
        const size_t done = chunks * sizeof(__m256i);
        if (done < n)
            std::memcpy(reinterpret_cast<char*>(dst) + done,
                        reinterpret_cast<const char*>(src) + done, n - done);
    }
}

namespace vulkanfs {
    namespace vulkan {

        VulkanState g_state;
        std::mutex g_vulkan_mutex;

        // True when blocks are DEVICE_LOCAL|HOST_VISIBLE (resizable BAR / SAM).
        // Set once on the first increase_pool() call; controls which I/O path is used.
        static bool g_host_accessible = false;

        // Each Segment is one 1 MiB Vulkan allocation, subdivided into sub-blocks.
        static constexpr size_t segment_size      = 1024 * 1024;
        static constexpr size_t blocks_per_segment = segment_size / block::size;

        // A Segment owns all Vulkan resources for a 1 MiB allocation and tracks
        // which sub-blocks are free via a bitmask (bit i = 1 → sub-block i is free).
        struct Segment {
            VkBuffer       buffer  = VK_NULL_HANDLE;
            VkDeviceMemory memory  = VK_NULL_HANDLE;
            void*          mapped  = nullptr;   // non-null only for host-accessible allocations

            // Staging path — only populated when !g_host_accessible:
            VkCommandPool   command_pool   = VK_NULL_HANDLE;
            VkFence         fence          = VK_NULL_HANDLE;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            VkBuffer        staging_buffer = VK_NULL_HANDLE;
            VkDeviceMemory  staging_memory = VK_NULL_HANDLE;
            void*           staging_mapped = nullptr;

            // Bitmask: bit i = 1 means sub-block i is free.
            // blocks_per_segment = 16, so uint32_t is sufficient.
            uint32_t free_bitmap = (1u << blocks_per_segment) - 1u;
            size_t   free_count  = blocks_per_segment;

            int allocate_sub_block() {
                if (free_count == 0) return -1;
                int idx = __builtin_ctz(free_bitmap);
                free_bitmap &= ~(1u << idx);
                free_count--;
                return idx;
            }

            void free_sub_block(size_t idx) {
                free_bitmap |= (1u << idx);
                free_count++;
            }
        };

        // All segments, allocated at startup and never freed during runtime.
        static std::vector<Segment*> segments;
        static int total_sub_blocks = 0;

        static size_t device_num = 0;

        static void check_vk_result(VkResult result, const char* op) {
            if (result != VK_SUCCESS)
                throw std::runtime_error(std::string("Vulkan error in ") + op + ": " + std::to_string(result));
        }

        static uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
            for (uint32_t i = 0; i < g_state.memory_properties.memoryTypeCount; i++) {
                if ((type_filter & (1 << i)) &&
                    (g_state.memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
                    return i;
            }
            throw std::runtime_error("Failed to find suitable memory type");
        }

        static bool init_vulkan() {
            if (g_state.initialized) return true;

            std::lock_guard<std::mutex> lock(g_vulkan_mutex);

            if (g_state.initialized) return true;

            try {
                VkApplicationInfo app_info{};
                app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
                app_info.pApplicationName = "VULKANFS";
                app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
                app_info.pEngineName = "No Engine";
                app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
                app_info.apiVersion = VK_API_VERSION_1_0;

                VkInstanceCreateInfo create_info{};
                create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
                create_info.pApplicationInfo = &app_info;

                check_vk_result(vkCreateInstance(&create_info, nullptr, &g_state.instance), "vkCreateInstance");

                uint32_t device_count = 0;
                vkEnumeratePhysicalDevices(g_state.instance, &device_count, nullptr);
                if (device_count == 0) {
                    std::cerr << "No Vulkan-capable GPU found" << std::endl;
                    return false;
                }

                std::vector<VkPhysicalDevice> devices(device_count);
                vkEnumeratePhysicalDevices(g_state.instance, &device_count, devices.data());

                if (device_num >= devices.size()) {
                    std::cerr << "Device index out of range" << std::endl;
                    return false;
                }

                g_state.physical_device = devices[device_num];

                vkGetPhysicalDeviceMemoryProperties(g_state.physical_device, &g_state.memory_properties);

                uint32_t queue_family_count = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(g_state.physical_device, &queue_family_count, nullptr);
                std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
                vkGetPhysicalDeviceQueueFamilyProperties(g_state.physical_device, &queue_family_count, queue_families.data());

                for (uint32_t i = 0; i < queue_family_count; i++) {
                    if (queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
                        g_state.transfer_queue_family = i;
                        break;
                    }
                }

                VkDeviceQueueCreateInfo queue_create_info{};
                queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queue_create_info.queueFamilyIndex = g_state.transfer_queue_family;
                queue_create_info.queueCount = 1;
                float queue_priority = 1.0f;
                queue_create_info.pQueuePriorities = &queue_priority;

                VkDeviceCreateInfo device_create_info{};
                device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
                device_create_info.pQueueCreateInfos = &queue_create_info;
                device_create_info.queueCreateInfoCount = 1;

                check_vk_result(vkCreateDevice(g_state.physical_device, &device_create_info, nullptr, &g_state.device), "vkCreateDevice");

                vkGetDeviceQueue(g_state.device, g_state.transfer_queue_family, 0, &g_state.transfer_queue);

                g_state.memory_type_index = find_memory_type(UINT32_MAX, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

                g_state.initialized = true;
                return true;

            } catch (const std::exception& e) {
                std::cerr << "Vulkan initialization failed: " << e.what() << std::endl;
                return false;
            }
        }

        bool is_available() {
            return init_vulkan();
        }

        void set_device(size_t device) {
            device_num = device;
        }

        std::vector<std::string> list_devices() {
            std::vector<std::string> device_names;

            VkInstance instance;
            VkApplicationInfo app_info{};
            app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            app_info.pApplicationName = "VULKANFS Query";
            app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            app_info.apiVersion = VK_API_VERSION_1_0;

            VkInstanceCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            create_info.pApplicationInfo = &app_info;

            if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS)
                return device_names;

            uint32_t device_count = 0;
            vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
            if (device_count > 0) {
                std::vector<VkPhysicalDevice> devices(device_count);
                vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

                for (const auto& device : devices) {
                    VkPhysicalDeviceProperties properties;
                    vkGetPhysicalDeviceProperties(device, &properties);
                    device_names.push_back(properties.deviceName);
                }
            }

            vkDestroyInstance(instance, nullptr);
            return device_names;
        }

        int pool_size() {
            return total_sub_blocks;
        }

        int pool_available() {
            int free = 0;
            for (const auto* seg : segments)
                free += (int)seg->free_count;
            return free;
        }

        // Create the staging buffer + command pool/buffer/fence for the non-BAR path.
        static void create_segment_staging(Segment* seg) {
            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool_info.queueFamilyIndex = g_state.transfer_queue_family;
            check_vk_result(vkCreateCommandPool(g_state.device, &pool_info, nullptr, &seg->command_pool), "vkCreateCommandPool");

            VkCommandBufferAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc_info.commandPool = seg->command_pool;
            alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc_info.commandBufferCount = 1;
            check_vk_result(vkAllocateCommandBuffers(g_state.device, &alloc_info, &seg->command_buffer), "vkAllocateCommandBuffers");

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            check_vk_result(vkCreateFence(g_state.device, &fence_info, nullptr, &seg->fence), "vkCreateFence");

            // Persistent host-visible staging buffer sized to one sub-block.
            // Only one sub-block is transferred at a time (serialised by fsmutex).
            VkBufferCreateInfo staging_info{};
            staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            staging_info.size = block::size;
            staging_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            staging_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            check_vk_result(vkCreateBuffer(g_state.device, &staging_info, nullptr, &seg->staging_buffer), "vkCreateBuffer (staging)");

            VkMemoryRequirements staging_req;
            vkGetBufferMemoryRequirements(g_state.device, seg->staging_buffer, &staging_req);

            VkMemoryAllocateInfo staging_alloc{};
            staging_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            staging_alloc.allocationSize = staging_req.size;
            staging_alloc.memoryTypeIndex = find_memory_type(staging_req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            check_vk_result(vkAllocateMemory(g_state.device, &staging_alloc, nullptr, &seg->staging_memory), "vkAllocateMemory (staging)");
            check_vk_result(vkBindBufferMemory(g_state.device, seg->staging_buffer, seg->staging_memory, 0), "vkBindBufferMemory (staging)");
            check_vk_result(vkMapMemory(g_state.device, seg->staging_memory, 0, block::size, 0, &seg->staging_mapped), "vkMapMemory (staging)");
        }

        size_t increase_pool(size_t size) {
            if (!init_vulkan()) return 0;

            int segment_count = (int)(1 + (size - 1) / segment_size);

            std::lock_guard<std::mutex> lock(g_vulkan_mutex);

            // On the first call, probe for resizable BAR (NVIDIA) / Smart Access Memory (AMD).
            if (total_sub_blocks == 0) {
                VkBufferCreateInfo probe_info{};
                probe_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                probe_info.size = segment_size;
                probe_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                probe_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                VkBuffer probe_buf;
                if (vkCreateBuffer(g_state.device, &probe_info, nullptr, &probe_buf) == VK_SUCCESS) {
                    VkMemoryRequirements probe_req;
                    vkGetBufferMemoryRequirements(g_state.device, probe_buf, &probe_req);
                    try {
                        find_memory_type(probe_req.memoryTypeBits,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                        g_host_accessible = true;
                    } catch (...) {}
                    vkDestroyBuffer(g_state.device, probe_buf, nullptr);
                }
                std::cout << (g_host_accessible
                    ? "host-accessible VRAM detected (BAR/SAM): using direct memcpy path"
                    : "no BAR/SAM: using staging buffer path") << std::endl;
            }

            const VkMemoryPropertyFlags mem_flags = g_host_accessible
                ? (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            for (int i = 0; i < segment_count; i++) {
                try {
                    Segment* seg = new Segment();

                    VkBufferCreateInfo buffer_info{};
                    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    buffer_info.size = segment_size;
                    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                    check_vk_result(vkCreateBuffer(g_state.device, &buffer_info, nullptr, &seg->buffer), "vkCreateBuffer");

                    VkMemoryRequirements mem_requirements;
                    vkGetBufferMemoryRequirements(g_state.device, seg->buffer, &mem_requirements);

                    VkMemoryAllocateInfo alloc_info{};
                    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                    alloc_info.allocationSize = mem_requirements.size;
                    alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, mem_flags);

                    check_vk_result(vkAllocateMemory(g_state.device, &alloc_info, nullptr, &seg->memory), "vkAllocateMemory");
                    check_vk_result(vkBindBufferMemory(g_state.device, seg->buffer, seg->memory, 0), "vkBindBufferMemory");

                    if (g_host_accessible)
                        check_vk_result(vkMapMemory(g_state.device, seg->memory, 0, segment_size, 0, &seg->mapped), "vkMapMemory (segment)");
                    else
                        create_segment_staging(seg);

                    segments.push_back(seg);
                    total_sub_blocks += (int)blocks_per_segment;

                } catch (const std::exception& e) {
                    std::cerr << "Failed to allocate segment: " << e.what() << std::endl;
                    return (size_t)i * segment_size;
                }
            }

            return (size_t)segment_count * segment_size;
        }

        block_ref allocate() {
            std::lock_guard<std::mutex> lock(g_vulkan_mutex);
            for (auto* seg : segments) {
                int idx = seg->allocate_sub_block();
                if (idx >= 0)
                    return block_ref(new block(seg, (size_t)idx));
            }
            return nullptr;
        }

        block::block(Segment* seg, size_t index)
            : _segment(seg), _sub_block_index(index) {}

        block::~block() {
            std::lock_guard<std::mutex> lock(g_vulkan_mutex);
            _segment->free_sub_block(_sub_block_index);
        }

        static void submit_command(Segment* seg, bool wait) {
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &seg->command_buffer;

            check_vk_result(vkQueueSubmit(g_state.transfer_queue, 1, &submit_info, seg->fence), "vkQueueSubmit");

            if (wait) {
                check_vk_result(vkWaitForFences(g_state.device, 1, &seg->fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
                check_vk_result(vkResetFences(g_state.device, 1, &seg->fence), "vkResetFences");
            }
        }

        void block::read(off_t offset, size_t size, void* data) const {
            if (dirty) {
                // Never written — return zeros to avoid leaking data from a prior use.
                memset(data, 0, size);
                return;
            }

            off_t abs_offset = (off_t)_sub_block_index * (off_t)block::size + offset;

            if (g_host_accessible) {
                memcpy(data, static_cast<const char*>(_segment->mapped) + abs_offset, size);
                return;
            }

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check_vk_result(vkResetCommandBuffer(_segment->command_buffer, 0), "vkResetCommandBuffer");
            check_vk_result(vkBeginCommandBuffer(_segment->command_buffer, &begin_info), "vkBeginCommandBuffer");

            VkBufferCopy copy_region{};
            copy_region.srcOffset = (VkDeviceSize)abs_offset;
            copy_region.dstOffset = 0;
            copy_region.size      = size;
            vkCmdCopyBuffer(_segment->command_buffer, _segment->buffer, _segment->staging_buffer, 1, &copy_region);

            check_vk_result(vkEndCommandBuffer(_segment->command_buffer), "vkEndCommandBuffer");
            submit_command(_segment, true);

            memcpy(data, _segment->staging_mapped, size);
        }

        void block::write(off_t offset, size_t size, const void* data, bool async) {
            (void)async;

            off_t abs_offset = (off_t)_sub_block_index * (off_t)block::size + offset;

            if (g_host_accessible) {
                memcpy_nt(static_cast<char*>(_segment->mapped) + abs_offset, data, size);
                dirty = false;
                return;
            }

            memcpy(_segment->staging_mapped, data, size);

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check_vk_result(vkResetCommandBuffer(_segment->command_buffer, 0), "vkResetCommandBuffer");
            check_vk_result(vkBeginCommandBuffer(_segment->command_buffer, &begin_info), "vkBeginCommandBuffer");

            VkBufferCopy copy_region{};
            copy_region.srcOffset = 0;
            copy_region.dstOffset = (VkDeviceSize)abs_offset;
            copy_region.size      = size;
            vkCmdCopyBuffer(_segment->command_buffer, _segment->staging_buffer, _segment->buffer, 1, &copy_region);

            check_vk_result(vkEndCommandBuffer(_segment->command_buffer), "vkEndCommandBuffer");
            submit_command(_segment, true);

            dirty = false;
        }

        void block::sync() {
            // Submissions are already synchronous (fenced), nothing to do.
        }
    }
}
