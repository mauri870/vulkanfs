#ifndef VULKANFS_UTIL_HPP
#define VULKANFS_UTIL_HPP

#define FUSE_USE_VERSION 30
#include <fuse.h>

#include <iostream>
#include <string>

using std::string;

namespace vulkanfs {
    namespace util {
        template<typename T>
        T fatal_error(const string& error, T ret) {
            std::cerr << "error: " << error << std::endl;
            fuse_exit(fuse_get_context()->fuse);
            return ret;
        }

        timespec time();

        // Splits "/path/to/file" into "/path/to" and "file". dir is "/" for top-level paths.
        void split_file_path(const string& path, string& dir, string& file);
    }
}

#endif
