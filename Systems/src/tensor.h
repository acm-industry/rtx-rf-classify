#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <memory>
#include <experimental/mdspan>
#include <concepts>
#include <algorithm>
#include <cassert>
#include <string>
#include "ExprSystem/Expression.h"

template<class E>
concept FixedExtent =
    requires {
        typename E::index_type;
        { E::rank_dynamic() } -> std::same_as<std::size_t>;
    } &&
    (E::rank_dynamic() == 0) &&
    (E::rank() > 0);


template<typename A, typename T>
concept Allocator =
    requires(A alloc, T* ptr, std::size_t n) {
        { alloc.allocate(n) } -> std::same_as<T*>;
        { alloc.deallocate(ptr, n) } -> std::same_as<void>;
    };

template<typename E>
constexpr std::size_t compute_static_size() {
    std::size_t result = 1;
    for (std::size_t i = 0; i < E::rank(); ++i) {
        result *= E::static_extent(i);
    }
    return result;
}

template <FixedExtent E>
constexpr std::array<std::size_t, E::rank()> compute_strides() {
    std::array<std::size_t, E::rank()> strides;
    std::size_t running = 1;
    for (std::size_t i = E::rank(); i-- > 0;) {
        strides[i] = running;
        running *= E::static_extent(i);
    }

    return strides;
}

template<std::size_t N, class E>
struct remove_extents;

// base case

template<class IndexType, std::size_t... Extents>
struct remove_extents<0, std::extents<IndexType, Extents...>> {
    using type = std::extents<IndexType, Extents...>;
};

// recursive case

template<std::size_t N, class IndexType, std::size_t First, std::size_t... Rest> requires ( N <= sizeof...(Rest) + 1 && N != 0 )
struct remove_extents<N, std::extents<IndexType, First, Rest...>> {
    using type = typename remove_extents<N - 1, std::extents<IndexType, Rest...>>::type;
};

template<std::size_t N, class E>
using remove_extents_t = typename remove_extents<N, E>::type;

/**
 * Tensor Base Class
 * Effectively acts as a std::mdspan using ONLY fixed/static extents, which means each Tensor is extremely cheap, at only 8 bytes per.
 * This allows comptime optimizations since the compiler can view a TensorBase as just a poiner with some indexing operations.
 * Ideally, this means loops like:
 * ``` for (auto row : TensorBase<float, std::extents<size_t, 4, 4>> ) { for (float& col : row) { col... } } ```
 * works at zero or near-zero cost. <-- yet to be verified
 */
template<class T, FixedExtent E> requires ( E::rank() > 0 )
class TensorBase {
public:
    using value_type = T;
    using element_type = T;
    using extents_type = E;
    using index_type = typename E::index_type;

    // Expression system contract.
    static constexpr auto extents = E{};
    static constexpr std::size_t rank = E::rank();
    static constexpr std::size_t static_size = compute_static_size<E>();
    static constexpr std::size_t iter_size() noexcept { return static_size; }

    static constexpr std::array<std::size_t, rank> strides = compute_strides<E>();

    static constexpr std::size_t static_extent(std::size_t i) { return E::static_extent(i); }

protected:
    T* data_;

    constexpr TensorBase() = default;

public:

    // Assumes valid pointer. Usage with nullptr is UB.
    template <std::size_t N> requires ( N == static_size )
    constexpr TensorBase(std::span<T, N> view) : data_( view.data() ) {}

    template <FixedExtent OtherExtent> requires ( static_size == TensorBase<T, OtherExtent>::static_size )
    constexpr TensorBase(const TensorBase<T, OtherExtent>& other) : data_( other.data() ) {}

    // Copy assign copies data, uses same pointer/memory
    template <FixedExtent OtherExtent> requires ( static_size == TensorBase<T, OtherExtent>::static_size )
    constexpr TensorBase& operator=(const TensorBase<T, OtherExtent>& other) {
        if constexpr ( extents == TensorBase<T, OtherExtent>::extents ) {
            if (this == &other) return *this;
        }
        std::copy_n( other.data(), static_size, data_ );
        return *this;
    }

    constexpr std::mdspan<T, E> view() noexcept { return std::mdspan<T,E>{data_}; }
    constexpr std::mdspan<const T, E> view() const noexcept { return std::mdspan<const T,E>{data_}; }

    constexpr TensorBase<T, std::extents<index_type, static_size>> flat_view() noexcept { return std::span<T, static_size>(data_, static_size); }
    constexpr TensorBase<const T, std::extents<index_type, static_size>> flat_view() const noexcept { return std::span<const T, static_size>(data_, static_size); }

    constexpr T& flat(std::size_t idx) {
        return data_[idx];
    }

    constexpr const T& flat(std::size_t idx) const {
        return data_[idx];
    }

    constexpr const value_type& access(std::size_t idx) const noexcept {
        return flat(idx);
    }

    constexpr value_type& access(std::size_t idx) noexcept {
        return flat(idx);
    }

    constexpr T* data() noexcept {
        return data_;
    }

    constexpr const T* data() const noexcept {
        return data_;
    }

    constexpr std::size_t size() const noexcept { return static_size; }
    constexpr std::size_t size(std::size_t idx) const noexcept { return E::static_extent(idx); }

    template <class... Idxs> requires ( sizeof...(Idxs) == rank && (std::integral<std::remove_cvref_t<Idxs>> && ...) )
    constexpr T& operator[](Idxs... idx) noexcept {
        std::size_t offset = 0;
        std::size_t i = 0;

        ((offset += static_cast<std::size_t>(idx) * strides[i++]), ...);

        return data_[offset];
    }

    template <class... Idxs> requires ( sizeof...(Idxs) == rank && (std::integral<std::remove_cvref_t<Idxs>> && ...) )
    constexpr const T& operator[](Idxs... idx) const noexcept {
        std::size_t offset = 0;
        std::size_t i = 0;

        ((offset += static_cast<std::size_t>(idx) * strides[i++]), ...);

        return data_[offset];
    }

    template <class... Idxs> requires ( sizeof...(Idxs) < rank && (std::integral<std::remove_cvref_t<Idxs>> && ...) )
    constexpr TensorBase<T, remove_extents_t<sizeof...(Idxs), E>> operator[](Idxs... idx) noexcept {
        using return_type = TensorBase<T, remove_extents_t<sizeof...(Idxs), E>>;

        std::size_t offset = 0;
        std::size_t i = 0;

        ((offset += static_cast<std::size_t>(idx) * strides[i++]), ...);


        return std::span< T, return_type::static_size >(data_ + offset, return_type::static_size);
    }

    template <class... Idxs> requires ( sizeof...(Idxs) < rank && (std::integral<std::remove_cvref_t<Idxs>> && ...) )
    constexpr const TensorBase<const T, remove_extents_t<sizeof...(Idxs), E>> operator[](Idxs... idx) const noexcept {
        using return_type = TensorBase<const T, remove_extents_t<sizeof...(Idxs), E>>;

        std::size_t offset = 0;
        std::size_t i = 0;

        ((offset += static_cast<std::size_t>(idx) * strides[i++]), ...);


        return std::span< const T, return_type::static_size >(data_ + offset, return_type::static_size);
    }

    class iterator {
        T* view;
        std::size_t idx;


    public:
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag; // Make bidirectional later?

        constexpr iterator(T* view = nullptr, std::size_t idx = 0) : view(view), idx(idx) {}

        constexpr decltype(auto) operator*() const noexcept {
            if constexpr (rank == 1) return view[idx];
            else {
                using return_type = TensorBase<T, remove_extents_t<1, E>>;
                return return_type(std::span<T, return_type::static_size>(view + idx * strides[0], return_type::static_size));
            }
        }

        using reference = decltype( std::declval<const iterator&>().operator*() );
        using value_type = std::remove_cvref_t<reference>;

        constexpr iterator& operator++() noexcept {
            ++idx;
            return *this;
        }
        
        constexpr iterator operator++(int) noexcept {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        friend constexpr bool operator==(const iterator& a, const iterator& b) noexcept {
            return a.view == b.view && a.idx == b.idx;
        }

        friend constexpr bool operator!=(const iterator& a, const iterator& b) noexcept {
            return a.view != b.view || a.idx != b.idx;
        }
    };

    constexpr iterator begin() noexcept {
        return iterator{ data_ };
    }

    constexpr iterator end() noexcept {
        return iterator{ data_, E::static_extent(0) };
    }

    std::string prettyPrint(std::size_t indent = 0) const {
        auto indent_str = [](std::size_t n) {
            return std::string(n, ' ');
        };

        if constexpr (rank == 1) {
            std::string out = "[";
            for (std::size_t i = 0; i < E::static_extent(0); ++i) {
                if (i) out += ", ";
                out += std::to_string((*this)[i]);
            }
            out += "]";
            return out;
        } else {
            std::string out = "[\n";
            for (std::size_t i = 0; i < E::static_extent(0); ++i) {
                out += indent_str(indent + 2);
                out += (*this)[i].prettyPrint(indent + 2);
                if (i + 1 < E::static_extent(0)) out += ",";
                out += "\n";
            }
            out += indent_str(indent);
            out += "]";
            return out;
        }
    }
    
};

// Dynamic Tensor capable of using an allocator to automatically manage memory RAII-style.
// It should be important to note that it does NOT call destructors on the value_type. This is not a vector class,
// so it would just be extra overhead.
// NOTE: The copy constructor is deleted; this is because of ambiguity between what it should do.
// Instead, a clone method is provided. For the purpose of Expressions, use the method .as_view(), which statically upcasts to a TensorBase.
template <class T, FixedExtent E, Allocator<T> A = std::allocator<T>>
class DynTensor : public TensorBase<T, E> {
    using Base = TensorBase<T, E>;

    [[no_unique_address]] A alloc_;

    void release_storage() noexcept {
        if (this->data_ == nullptr) return;
        alloc_.deallocate(this->data_, Base::static_size);
        this->data_ = nullptr;
    }

    template <class X, FixedExtent Y, Allocator<X> Z>
    friend class DynTensor;

public:
    ~DynTensor() {
        release_storage();
    }

    
    Base& as_view() noexcept { return static_cast<Base&>(*this); }

    // Creates a copy of arbitrary shape. Can be used as a reshape, but a more performant version should be considered in
    // doing something like DynTensor<T, new_shape, A> new_owner = std::move( this_tensor );
    template <FixedExtent Extent = typename Base::extents_type> requires ( Base::static_size == TensorBase<T, Extent>::static_size )
    DynTensor<T, Extent, A> clone() {
        DynTensor<T, Extent, A> ret(alloc_);
        std::copy_n( this->data_, Base::static_size, ret.data_ );
        return ret;
    }

    DynTensor(const A& allocator = A()) noexcept : Base(), alloc_(allocator) {
        this->data_ = alloc_.allocate(Base::static_size);
    }

    DynTensor(const DynTensor& other) = delete;

    template <FixedExtent OtherExtent, Allocator<T> OtherA> requires ( Base::static_size == TensorBase<T, OtherExtent>::static_size )
    constexpr DynTensor& operator=(const DynTensor<T, OtherExtent, OtherA>& other) {
        return static_cast<DynTensor&>(Base::operator=( other ));
    }

    template <Expression Expr> requires ( Expr::iter_size() == Base::static_size )
    constexpr DynTensor(const Expr& expr, const A& allocator = A()) noexcept : Base(), alloc_(allocator) {
        this->data_ = alloc_.allocate(Base::static_size);
        for (size_t i = 0; i < Base::static_size; i++) this->data_[i] = expr.access(i);
    }

    template <Expression Expr> requires ( Expr::iter_size() == Base::static_size )
    constexpr DynTensor& operator=(const Expr& expr) {
        for (size_t i = 0; i < Base::static_size; i++) this->data_[i] = expr.access(i);
        return *this;
    }

    template <FixedExtent OtherExtent> requires ( Base::static_size == TensorBase<T, OtherExtent>::static_size )
    DynTensor(DynTensor<T, OtherExtent, A>&& other) noexcept : 
        Base( std::span<T, Base::static_size>( other.data_, Base::static_size ) ), 
        alloc_( std::move(other.alloc_) )
    {
        other.data_ = nullptr;
    }
    
    template <FixedExtent OtherExtent> requires ( Base::static_size == TensorBase<T, OtherExtent>::static_size )
    DynTensor& operator=(DynTensor<T, OtherExtent, A>&& other) noexcept {
        if (this == &other) return *this;

        release_storage();
        alloc_ = std::move(other.alloc_);
        this->data_ = other.data_;
        other.data_ = nullptr;
        return *this;
    }
};

#endif
