# vulkanFS - A Vulkan-based filesystem

This project implements a FUSE filesystem that stores and manages files directly in GPU VRAM using Vulkan

Why? Because it's an interesting idea. Perhaps someone finds real use for it.

## Requirements

- Vulkan SDK and drivers
- FUSE 3.0+
- C++20 compatible compiler

## Building

```bash
make
```

## Usage

```bash
# Create mount point
mkdir -p /tmp/vulkanfs

# Mount a 256mb vram filesystem in /tmp/vulkanfs
./bin/vulkanfs /tmp/vulkanfs 256MB

echo 'This lives in VRAM!' > /tmp/vulkanfs/file.txt
cat /tmp/vulkanfs/file.txt

# Unmount
fusermount -u /tmp/vulkanfs
```
