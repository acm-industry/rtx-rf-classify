#include <cstddef>
#include <functional>
#include <span>

class MemoryBuffer {
   private:
    std::byte* buf;
    std::size_t capacity;
    std::size_t offset;
    bool owns;

    void* alloc(std::size_t num);

   public:
    MemoryBuffer(std::size_t capacity)
        : buf(new std::byte[capacity]),
          capacity(capacity),
          offset(0),
          owns(true) {}
    MemoryBuffer(std::span<std::byte> ext)
        : buf(ext.data()), capacity(ext.size()), offset(0), owns(false) {}
    ~MemoryBuffer();

    MemoryBuffer(const MemoryBuffer&) = delete;
    MemoryBuffer& operator=(const MemoryBuffer&) = delete;
    MemoryBuffer(MemoryBuffer&& other) = delete;
    MemoryBuffer& operator=(MemoryBuffer&& other) = delete;

    template <class T>
    class Allocator {
       private:
        friend class MemoryBuffer;

        std::reference_wrapper<MemoryBuffer> buffer;

        Allocator(MemoryBuffer& buf) : buffer(buf) {}

       public:
        using value_type = T;

        T* allocate(std::size_t n) {
            return static_cast<T*>(buffer.get().alloc(n * sizeof(T)));
        }

        void deallocate(T* p, std::size_t n) {}  // no-op for bump allocator
    };

    template <class T>
    Allocator<T> get_allocator() {
        return Allocator<T>(*this);
    }
};