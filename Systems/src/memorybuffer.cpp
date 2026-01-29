#include "memorybuffer.h"

MemoryBuffer::~MemoryBuffer() {
    if (owns) delete[] buf;
}

void* MemoryBuffer::alloc(std::size_t num) {
    if (offset + num > capacity) throw std::bad_alloc();
    void* ptr = buf + offset;
    offset += num;
    return ptr;
}