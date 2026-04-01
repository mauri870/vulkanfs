#include "entry.hpp"
#include "util.hpp"
#include <cstring>

namespace vulkanfs {
    namespace entry {
        file_ref file_t::make(dir_ptr parent, const string& name) {
            auto file = file_ref(new file_t());
            file->link(parent, name);
            return file;
        }

        file_t::file_t() {
            mode(0644);
        }

        type::type_t file_t::type() const {
            return type::file;
        }

        size_t file_t::size() const {
            return _size;
        }

        void file_t::size(size_t new_size) {
            if (new_size < _size)
                free_blocks(new_size);

            _size = new_size;

            mtime(util::time());
        }

        int file_t::read(off_t off, size_t size, char* data, std::mutex& wait_mutex) {
            if ((size_t) off >= _size) return 0;
            size = std::min(_size - off, size);

            off_t end_pos = off + size;
            size_t total_read = size;

            while (off < end_pos) {
                off_t block_start = (off / vulkanfs::block::size) * vulkanfs::block::size;
                off_t block_off = off - block_start;
                size_t read_size = std::min(vulkanfs::block::size - block_off, size);

                auto block = get_block(block_start);

                // Drop the lock while the GPU transfer completes so other threads can proceed.
                wait_mutex.unlock();
                if (block) {
                    block->read(block_off, read_size, data);
                } else {
                    memset(data, 0, read_size);
                }
                wait_mutex.lock();

                data += read_size;
                off += read_size;
                size -= read_size;
            }

            atime(util::time());

            return total_read;
        }

        int file_t::write(off_t off, size_t size, const char* data, bool async) {
            off_t end_pos = off + size;
            size_t total_write = size;

            while (off < end_pos) {
                off_t block_start = (off / vulkanfs::block::size) * vulkanfs::block::size;
                off_t block_off = off - block_start;
                size_t write_size = std::min(vulkanfs::block::size - block_off, size);

                auto block = get_block(block_start);

                if (!block) {
                    block = alloc_block(block_start);

                    if (!block) break; // out of VRAM
                }

                block->write(block_off, write_size, data, async);

                last_written_block = block;

                data += write_size;
                off += write_size;
                size -= write_size;
            }

            if (_size < (size_t) off)
                _size = off;
            mtime(util::time());

            if (off < end_pos)
                return -ENOSPC;
            else
                return total_write;
        }

        void file_t::sync() {
            if (last_written_block)
                last_written_block->sync();
        }

        vulkanfs::block_ref file_t::get_block(off_t off) const {
            auto it = file_blocks.find(off);

            if (it != file_blocks.end())
                return it->second;
            else
                return nullptr;
        }

        vulkanfs::block_ref file_t::alloc_block(off_t off) {
            auto block = vulkanfs::allocate();

            if (block)
                file_blocks[off] = block;

            return block;
        }

        void file_t::free_blocks(off_t off) {
            // Round up to the next block boundary so the block containing *off* is kept.
            off_t start_off = (off / vulkanfs::block::size) * vulkanfs::block::size;
            if (off % vulkanfs::block::size != 0) start_off += vulkanfs::block::size;

            for (auto it = file_blocks.lower_bound(start_off); it != file_blocks.end();) {
                if (it->second == last_written_block)
                    last_written_block = nullptr;
                it = file_blocks.erase(it);
            }
        }
    }
}
