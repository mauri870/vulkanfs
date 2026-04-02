CXX ?= g++
CXXFLAGS += -std=c++20 $(shell pkg-config fuse3 --cflags) -I include/ -mavx2 -O2
LDFLAGS += $(shell pkg-config fuse3 --libs) -lvulkan
SOURCES = src/util.cpp src/vulkan_memory.cpp src/entry.cpp src/file.cpp src/dir.cpp src/symlink.cpp src/vulkanfs.cpp

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -DDEBUG -Wall -Werror
endif

OBJECTS = $(SOURCES:src/%.cpp=build/%.o)

bin/vulkanfs: $(OBJECTS) | bin
	$(CXX) -o $@ $^ $(LDFLAGS)

build bin:
	@mkdir -p $@

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean clean-all
clean:
	rm -rf build/ bin/

clean-all: clean
