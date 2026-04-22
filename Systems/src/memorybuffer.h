#ifndef __MEMORY_BUFFER_H__
#define __MEMORY_BUFFER_H__

#include <cstddef>
#include <functional>
#include <span>
#include <cstdlib>
#include <memory>

// Bare-metal targets compile with -fno-exceptions and have nowhere to surface
// allocator failures - we abort instead.  The native build keeps the
// std::bad_alloc semantics the rest of the codebase already expects.
#if defined(__cpp_exceptions) && __cpp_exceptions
#  define MEMBUF_OOM() throw std::bad_alloc()
#else
#  define MEMBUF_OOM() std::abort()
#endif

class MemoryBuffer {
   private:
    std::byte* buf;
    std::size_t capacity;
    std::size_t offset;
    bool owns;

    template <class T, std::size_t Alignment = alignof(T)> requires ( Alignment >= alignof(T) && ( ( Alignment & (Alignment - 1) ) == 0 ) )
    T* alloc(std::size_t num) {
        if (num == 0) MEMBUF_OOM();

        std::size_t bytes = sizeof(T) * num;
        std::byte* current = buf + offset;
        std::size_t space = capacity - offset;
        void* aligned_ptr = current;

        if ( !std::align( Alignment, bytes, aligned_ptr, space ) ) MEMBUF_OOM();

        std::byte* aligned_byte_ptr = static_cast<std::byte*>(aligned_ptr);
        offset = ( aligned_byte_ptr - buf ) + bytes;

        return reinterpret_cast<T*>(aligned_ptr);
    }

   public:
    MemoryBuffer(std::size_t capacity)
        : buf( (std::byte*)std::aligned_alloc(32, capacity) ),
          capacity(capacity),
          offset(0),
          owns(true) {}

    MemoryBuffer(std::span<std::byte> ext)
        : buf(ext.data()), capacity(ext.size()), offset(0), owns(false) {}

    ~MemoryBuffer() {
        if (owns) std::free(buf);
    }

    void reset() {
        offset = 0;
    }

    MemoryBuffer(const MemoryBuffer&) = delete;
    MemoryBuffer& operator=(const MemoryBuffer&) = delete;
    MemoryBuffer(MemoryBuffer&& other) = delete;
    MemoryBuffer& operator=(MemoryBuffer&& other) = delete;

    template <class T, std::size_t Alignment = alignof(T)>
    class Allocator {
       private:
        friend class MemoryBuffer;

        std::reference_wrapper<MemoryBuffer> buffer;

        Allocator(MemoryBuffer& buf) : buffer(buf) {}

       public:
        using value_type = T;
        using propagate_on_container_copy_assignment = std::true_type;
        using propagate_on_container_move_assignment = std::true_type;
        using propagate_on_container_swap = std::true_type;
        using is_always_equal = std::false_type;

        template <class U, std::size_t UAlignment = Alignment>
        Allocator(const Allocator<U, UAlignment>& other) : buffer(other.buffer.get()) {}
        template<class U, std::size_t UAlignment = Alignment> struct rebind { using other = Allocator<U, UAlignment>; };

        T* allocate(std::size_t n) {
            return buffer.get().template alloc<T, Alignment>(n);
        }

        void deallocate([[maybe_unused]] T* p, [[maybe_unused]] std::size_t n) {}  // no-op for bump allocator

        bool operator==(const Allocator& other) const {
            return &buffer.get() == &other.buffer.get();
        }

        bool operator!=(const Allocator& other) const {
            return !(*this == other);
        }
    };

    template <class T, std::size_t Alignment = alignof(T)>
    Allocator<T, Alignment> get_allocator() {
        return Allocator<T, Alignment>(*this);
    }
};

#endif