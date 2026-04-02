#!/bin/bash
set -e

MOUNTPOINT=/tmp/vram
VRAM_SIZE=2GB
TEST_FILE="$MOUNTPOINT/benchmark.bin"

echo "==> Building..."
make -j$(nproc)

echo "==> Mounting vulkanfs at $MOUNTPOINT (${VRAM_SIZE})..."
mkdir -p "$MOUNTPOINT"
bin/vulkanfs "$MOUNTPOINT" "$VRAM_SIZE" &
FUSE_PID=$!

# Wait for the filesystem to be ready
sleep 1

cleanup() {
    echo "==> Unmounting..."
    fusermount -u "$MOUNTPOINT" && echo "Unmounted $MOUNTPOINT"
}
trap cleanup EXIT

echo "==> Writing 1G file with dd..."
dd if=/dev/zero of="$TEST_FILE" bs=1M count=1024 status=progress

echo "==> Benchmark complete."
