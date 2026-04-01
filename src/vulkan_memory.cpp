#include "vulkan_memory.hpp"
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace vulkanfs {
    namespace vulkan {

        VulkanState g_state;
        std::mutex g_vulkan_mutex;

        // True when blocks are DEVICE_LOCAL|HOST_VISIBLE (resizable BAR / SAM).
        // Set once on the first increase_pool() call; controls which I/O path is used.
        bool g_host_accessible = false;

        struct PoolEntry {
            VkBuffer       buffer;
            VkDeviceMemory memory;
            void*          mapped; // non-null only for host-accessible allocations
        };
        std::vector<PoolEntry> pool;
        int total_blocks = 0;

        size_t device_num = 0;

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
            return total_blocks;
        }

        int pool_available() {
            return pool.size();
        }

        size_t increase_pool(size_t size) {
            if (!init_vulkan()) return 0;

            int block_count = 1 + (size - 1) / block::size;

            std::lock_guard<std::mutex> lock(g_vulkan_mutex);

            // On the first call, probe for resizable BAR (NVIDIA) / Smart Access Memory (AMD).
            // If the GPU exposes DEVICE_LOCAL|HOST_VISIBLE memory we can skip the staging
            // buffer entirely and just memcpy directly into VRAM.
            if (total_blocks == 0) {
                VkBufferCreateInfo probe_info{};
                probe_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                probe_info.size = block::size;
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

            for (int i = 0; i < block_count; i++) {
                try {
                    VkBufferCreateInfo buffer_info{};
                    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    buffer_info.size = block::size;
                    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                    VkBuffer buffer;
                    check_vk_result(vkCreateBuffer(g_state.device, &buffer_info, nullptr, &buffer), "vkCreateBuffer");

                    VkMemoryRequirements mem_requirements;
                    vkGetBufferMemoryRequirements(g_state.device, buffer, &mem_requirements);

                    VkMemoryAllocateInfo alloc_info{};
                    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                    alloc_info.allocationSize = mem_requirements.size;
                    alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, mem_flags);

                    VkDeviceMemory memory;
                    check_vk_result(vkAllocateMemory(g_state.device, &alloc_info, nullptr, &memory), "vkAllocateMemory");
                    check_vk_result(vkBindBufferMemory(g_state.device, buffer, memory, 0), "vkBindBufferMemory");

                    void* mapped = nullptr;
                    if (g_host_accessible)
                        check_vk_result(vkMapMemory(g_state.device, memory, 0, block::size, 0, &mapped), "vkMapMemory (pool)");

                    pool.push_back({buffer, memory, mapped});
                    total_blocks++;

                } catch (const std::exception& e) {
                    std::cerr << "Failed to allocate block: " << e.what() << std::endl;
                    return i * block::size;
                }
            }

            return block_count * block::size;
        }

        block_ref allocate() {
            std::lock_guard<std::mutex> lock(g_vulkan_mutex);
            if (!pool.empty())
                return block_ref(new block());
            return nullptr;
        }

        block::block() {
            if (pool.empty())
                throw std::runtime_error("No blocks available in pool");

            auto entry = pool.back();
            pool.pop_back();

            buffer          = entry.buffer;
            memory          = entry.memory;
            host_accessible = g_host_accessible;
            mapped_ptr      = entry.mapped;

            if (!host_accessible)
                create_command_buffer();
        }

        block::~block() {
            std::lock_guard<std::mutex> lock(g_vulkan_mutex);
            pool.push_back({buffer, memory, mapped_ptr});

            if (!host_accessible) {
                vkUnmapMemory(g_state.device, staging_memory);
                vkDestroyBuffer(g_state.device, staging_buffer, nullptr);
                vkFreeMemory(g_state.device, staging_memory, nullptr);
                vkDestroyCommandPool(g_state.device, command_pool, nullptr); // frees command_buffer implicitly
                vkDestroyFence(g_state.device, fence, nullptr);
            }
        }

        void block::create_command_buffer() {
            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool_info.queueFamilyIndex = g_state.transfer_queue_family;
            check_vk_result(vkCreateCommandPool(g_state.device, &pool_info, nullptr, &command_pool), "vkCreateCommandPool");

            VkCommandBufferAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc_info.commandPool = command_pool;
            alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc_info.commandBufferCount = 1;
            check_vk_result(vkAllocateCommandBuffers(g_state.device, &alloc_info, &command_buffer), "vkAllocateCommandBuffers");

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            check_vk_result(vkCreateFence(g_state.device, &fence_info, nullptr, &fence), "vkCreateFence");

            // Persistent host-visible staging buffer, kept permanently mapped.
            // Avoids a vkAllocateMemory round-trip on every read/write.
            VkBufferCreateInfo staging_info{};
            staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            staging_info.size = block::size;
            staging_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            staging_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            check_vk_result(vkCreateBuffer(g_state.device, &staging_info, nullptr, &staging_buffer), "vkCreateBuffer (staging)");

            VkMemoryRequirements staging_req;
            vkGetBufferMemoryRequirements(g_state.device, staging_buffer, &staging_req);

            VkMemoryAllocateInfo staging_alloc{};
            staging_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            staging_alloc.allocationSize = staging_req.size;
            staging_alloc.memoryTypeIndex = find_memory_type(staging_req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            check_vk_result(vkAllocateMemory(g_state.device, &staging_alloc, nullptr, &staging_memory), "vkAllocateMemory (staging)");
            check_vk_result(vkBindBufferMemory(g_state.device, staging_buffer, staging_memory, 0), "vkBindBufferMemory (staging)");
            check_vk_result(vkMapMemory(g_state.device, staging_memory, 0, block::size, 0, &staging_mapped), "vkMapMemory (staging)");
        }

        void block::submit_command(VkCommandBuffer cmd, bool wait) {
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &cmd;

            check_vk_result(vkQueueSubmit(g_state.transfer_queue, 1, &submit_info, fence), "vkQueueSubmit");

            if (wait) {
                check_vk_result(vkWaitForFences(g_state.device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
                check_vk_result(vkResetFences(g_state.device, 1, &fence), "vkResetFences");
            }
        }

        void block::read(off_t offset, size_t size, void* data) const {
            if (dirty) {
                // Never written — return zeros rather than leaking data from a prior pool user.
                memset(data, 0, size);
                return;
            }

            if (host_accessible) {
                memcpy(data, static_cast<const char*>(mapped_ptr) + offset, size);
                return;
            }

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check_vk_result(vkResetCommandBuffer(command_buffer, 0), "vkResetCommandBuffer");
            check_vk_result(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");

            VkBufferCopy copy_region{};
            copy_region.srcOffset = offset;
            copy_region.dstOffset = 0;
            copy_region.size = size;
            vkCmdCopyBuffer(command_buffer, buffer, staging_buffer, 1, &copy_region);

            check_vk_result(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
            const_cast<block*>(this)->submit_command(command_buffer, true);

            memcpy(data, staging_mapped, size);
        }

        void block::write(off_t offset, size_t size, const void* data, bool async) {
            (void) async;

            if (host_accessible) {
                memcpy(static_cast<char*>(mapped_ptr) + offset, data, size);
                dirty = false;
                return;
            }

            memcpy(staging_mapped, data, size);

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check_vk_result(vkResetCommandBuffer(command_buffer, 0), "vkResetCommandBuffer");
            check_vk_result(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");

            VkBufferCopy copy_region{};
            copy_region.srcOffset = 0;
            copy_region.dstOffset = offset;
            copy_region.size = size;
            vkCmdCopyBuffer(command_buffer, staging_buffer, buffer, 1, &copy_region);

            check_vk_result(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
            submit_command(command_buffer, true);

            dirty = false;
        }

        void block::sync() {
            // Submissions are already synchronous (fenced), nothing to do.
        }
    }
}
