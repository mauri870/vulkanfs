#ifndef VULKANFS_VULKAN_MEMORY_HPP
#define VULKANFS_VULKAN_MEMORY_HPP

#include <vulkan/vulkan.h>
#include <memory>
#include <vector>
#include <mutex>

namespace vulkanfs {
    namespace vulkan {
        // Forward declaration — defined in vulkan_memory.cpp
        struct Segment;

        class block;
        typedef std::shared_ptr<block> block_ref;

        // Check if current machine supports VRAM allocation
        bool is_available();

        // Set the device to use
        void set_device(size_t num);

        // Returns a list of device names
        std::vector<std::string> list_devices();

        // Total blocks and blocks currently free (in block::size units)
        int pool_size();
        int pool_available();

        // Allocate pool of memory blocks, returns actual amount allocated (in bytes)
        size_t increase_pool(size_t size);

        // Get a new block of memory from the pool, returns nullptr if pool is empty
        block_ref allocate();

        /*
         * A logical sub-block within a VRAM segment.
         *
         * Each Vulkan allocation is 1 MiB (the Segment). A Segment is subdivided
         * into (1 MiB / block::size) sub-blocks. A block is a handle to one
         * sub-block; the underlying Vulkan resources are shared through Segment.
         */
        class block : public std::enable_shared_from_this<block> {
            friend block_ref allocate();

        public:
            // Logical block size (sub-block granularity, also the FUSE max_write size)
            static const size_t size = 64 * 1024;

            block(const block& other) = delete;

            ~block();

            void read(off_t offset, size_t size, void* data) const;

            // Data may be freed afterwards, even if called with async = true
            void write(off_t offset, size_t size, const void* data, bool async = false);

            // Wait for all writes to this block to complete
            void sync();

        private:
            Segment* _segment;
            size_t   _sub_block_index;

            // True until the first write in this allocation lifetime.
            // Reads before any write return zeros rather than leaking old data.
            bool dirty = true;

            block(Segment* seg, size_t index);
        };

        // Global Vulkan state
        struct VulkanState {
            VkInstance instance;
            VkPhysicalDevice physical_device;
            VkDevice device;
            VkQueue transfer_queue;
            uint32_t transfer_queue_family;

            VkPhysicalDeviceMemoryProperties memory_properties;
            uint32_t memory_type_index;

            bool initialized = false;
        };

        extern VulkanState g_state;
        extern std::mutex g_vulkan_mutex;
    }

    // Bring Vulkan symbols into the vulkanfs namespace
    using vulkan::block;
    using vulkan::block_ref;
    using vulkan::is_available;
    using vulkan::set_device;
    using vulkan::list_devices;
    using vulkan::pool_size;
    using vulkan::pool_available;
    using vulkan::increase_pool;
    using vulkan::allocate;
}

#endif
