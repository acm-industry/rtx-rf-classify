#include <memory>
#include <mdspan>
#include <concepts>
#include <algorithm>
#include <stdexcept>

template<typename T>
concept FixedExtent = 
    requires { typename T::index_type; } &&
    std::unsigned_integral<typename T::index_type> &&
    requires(T t) {t.rank(); } &&
    ([]<std::size_t... Is>(std::index_sequence<Is...>) {
        return ((T::static_extent(Is) != std::dynamic_extent) && ...);
    })(std::make_index_sequence<T::rank()>{});

template<typename A, typename T>
concept Allocator = 
    requires(A alloc, T* ptr, std::size_t n) {
        { alloc.allocate(n) } -> std::same_as<T*>;
        { alloc.deallocate(ptr, n) } -> std::same_as<void>;
    };

// compute size of matrix
template<typename E>
constexpr std::size_t compute_static_size() {
    std::size_t result = 1;
    for (std::size_t i = 0; i < E::rank(); ++i) {
        result *= E::static_extent(i);
    }
    return result;
}

template<typename T, typename Allocator>
struct AllocatorDeleter {
    Allocator alloc;
    std::size_t size;
    
    void operator()(T* ptr) const {
        if (ptr) {
            Allocator a = alloc;
            a.deallocate(ptr, size);
        }
    }
};

template<std::floating_point T, FixedExtent E, Allocator<T> A = std::allocator<T>>
class TensorBase {
public:
    using value_type = T;
    using extents_type = E;
    using allocator_type = A;
    using mdspan_type = std::mdspan<T, E>;
    using index_type = typename E::index_type;
    using deleter_type = AllocatorDeleter<T, A>;

    static constexpr std::size_t rank = E::rank();
    static constexpr std::size_t size = compute_static_size<E>();

private:
    mdspan_type data_;
    std::unique_ptr<T[], deleter_type> owned_;
    allocator_type allocator_;

public:
    // constructor
    TensorBase() : TensorBase(T{}) {}
    
    explicit TensorBase(const T* model_weights) 
        : data_(nullptr, E{}),
        owned_(nullptr, deleter_type{allocator_, size}),
        allocator_() {
        allocate_and_copy(model_weights);
    }

    // rule of 5
    ~TensorBase() = default;

    TensorBase(const TensorBase& other) 
        : data_(nullptr, E{}), 
          owned_(nullptr, deleter_type{other.allocator_, size}), 
          allocator_(other.allocator_) {
        allocate_and_copy(other.data_.data_handle());
    }
    
    TensorBase& operator=(const TensorBase& other) {
        if (this != &other) {
            allocator_ = other.allocator_;
            allocate_and_copy(other.data_.data_handle());
        }
        return *this;
    }

    TensorBase(TensorBase&& other) noexcept
        : data_(other.data_), 
          owned_(std::move(other.owned_)), 
          allocator_(std::move(other.allocator_)) {
        other.data_ = mdspan_type(nullptr, E{});
    }

    TensorBase& operator=(TensorBase&& other) noexcept {
        if (this != &other) {
            owned_ = std::move(other.owned_);
            data_ = other.data_;
            allocator_ = std::move(other.allocator_);
            other.data_ = mdspan_type(nullptr, E{});
        }
        return *this;
    }

    // accessors
    mdspan_type& view() noexcept { return data_; }
    const mdspan_type& view() const noexcept { return data_; }

    template<typename... Idx>
    requires(sizeof...(Idx) == rank)
    T& operator()(Idx... idx) { return data_[idx...]; }

    template<typename... Idx>
    requires(sizeof...(Idx) == rank)
    const T& operator()(Idx... idx) const { return data_[idx...]; }

    T* data() noexcept { return data_.data_handle(); }
    const T* data() const noexcept { return data_.data_handle(); }

    T& flat(std::size_t idx) {
        return data_.data_handle()[idx];
    }

private:
    void allocate_and_copy(const T* src) {
        T* ptr = allocator_.allocate(size);
        owned_ = std::unique_ptr<T[], deleter_type>(ptr, deleter_type{allocator_, size});
        std::copy_n(src, size, ptr);
        data_ = mdspan_type(owned_.get(), E{});
    }
};