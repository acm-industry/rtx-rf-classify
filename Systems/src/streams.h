#ifndef __STREAMS_H__
#define __STREAMS_H__

#include <concepts>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <type_traits>

// Exception support is optional: bare-metal newlib targets (e.g. Ara) compile
// with -fno-exceptions, in which case we abort instead of throwing.  Keeping
// the macro here means the rest of the file reads as plain `STREAM_FAIL(msg)`
// regardless of build flavour.
#if defined(__cpp_exceptions) && __cpp_exceptions
#  include <stdexcept>
#  define STREAM_FAIL(msg) throw std::runtime_error(msg)
#else
#  include <cstdlib>
#  define STREAM_FAIL(msg) std::abort()
#endif

#include "tensor.h"

/**
 * CRTP I/O stream layer.
 *
 * The base classes (IStream / OStream / IOStream) own a buffer of static size
 * BufferSize allocated through the supplied Allocator and provide buffered
 * byte-level transfer plus generic operator>> / operator<< for any trivially
 * copyable type and for TensorBase<T, E> instances.
 *
 * The Derived class is required to implement raw byte-level transport:
 *   std::size_t read_some (std::span<std::byte>       dst);   // for IStream
 *   std::size_t write_some(std::span<const std::byte> src);   // for OStream
 *
 * read_some returns 0 to signal EOF / closed peer; non-zero is the number of
 * bytes actually transferred (may be less than dst.size()).  The base loops as
 * needed.  write_some has the same semantics; flush() drains the buffer.
 *
 * The expected-method requirements are encoded as concepts (ByteSource /
 * ByteSink) so a violating Derived produces one targeted compile error rather
 * than a wall of template instantiation noise.
 *
 * Endianness is the caller's problem: trivially-copyable << / >> just memcpy
 * the raw bytes.  Wire formats that need network byte order should call
 * htonl / ntohl explicitly at the protocol layer (matches existing MVP).
 */

// ────────────────────────────────────────────────────────────────────────────
//  Concepts: required Derived methods
// ────────────────────────────────────────────────────────────────────────────

template <class D>
concept ByteSource = requires (D& d, std::span<std::byte> s) {
    { d.read_some(s) } -> std::convertible_to<std::size_t>;
};

template <class D>
concept ByteSink = requires (D& d, std::span<const std::byte> s) {
    { d.write_some(s) } -> std::convertible_to<std::size_t>;
};

// Optional: Derived may provide sync() to flush any underlying transport
// after the OStream buffer has been drained (e.g. force a TCP push, fsync).
template <class D>
concept HasSync = requires (D& d) {
    { d.sync() };
};

// ────────────────────────────────────────────────────────────────────────────
//  IStream: buffered byte-source
// ────────────────────────────────────────────────────────────────────────────

template <std::size_t BufferSize, class Allocator, class Derived>
class IStream {
    static_assert(BufferSize > 0, "IStream BufferSize must be > 0");

    using ByteAlloc = typename std::allocator_traits<Allocator>
                          ::template rebind_alloc<std::byte>;
    using ByteAllocTraits = std::allocator_traits<ByteAlloc>;

    ByteAlloc alloc_;
    std::byte* buf_;
    std::byte* head_;  // next byte to read out
    std::byte* tail_;  // one-past-last valid byte

    Derived&       self()       noexcept { return static_cast<Derived&>(*this); }
    const Derived& self() const noexcept { return static_cast<const Derived&>(*this); }

    // Move all unread data to the start of the buffer, then ask Derived to
    // top us up.  Returns bytes added (0 == EOF on this attempt).
    std::size_t refill() {
        if (head_ != buf_) {
            std::size_t carry = static_cast<std::size_t>(tail_ - head_);
            if (carry > 0) std::memmove(buf_, head_, carry);
            head_ = buf_;
            tail_ = buf_ + carry;
        }
        std::span<std::byte> dst{ tail_, static_cast<std::size_t>((buf_ + BufferSize) - tail_) };
        if (dst.empty()) return 0;
        std::size_t got = self().read_some(dst);
        tail_ += got;
        return got;
    }

   public:
    explicit IStream(const Allocator& alloc = Allocator{})
        : alloc_(alloc),
          buf_(ByteAllocTraits::allocate(alloc_, BufferSize)),
          head_(buf_),
          tail_(buf_) {}

    ~IStream() {
        // Deferred until Derived is complete: enforces the CRTP contract.
        static_assert(ByteSource<Derived>,
            "IStream<...> Derived must implement: "
            "std::size_t read_some(std::span<std::byte>)");
        if (buf_) ByteAllocTraits::deallocate(alloc_, buf_, BufferSize);
    }

    IStream(const IStream&) = delete;
    IStream& operator=(const IStream&) = delete;
    IStream(IStream&&) = delete;
    IStream& operator=(IStream&&) = delete;

    static constexpr std::size_t buffer_size() noexcept { return BufferSize; }
    std::size_t available() const noexcept { return static_cast<std::size_t>(tail_ - head_); }

    // Read exactly dst.size() bytes, refilling from Derived as needed.
    // Throws std::runtime_error on EOF before satisfying the request.
    void read_exact(std::span<std::byte> dst) {
        std::byte* out = dst.data();
        std::size_t remaining = dst.size();

        while (remaining > 0) {
            std::size_t buffered = available();

            if (buffered >= remaining) {
                std::memcpy(out, head_, remaining);
                head_ += remaining;
                return;
            }

            if (buffered > 0) {
                std::memcpy(out, head_, buffered);
                out += buffered;
                remaining -= buffered;
                head_ = tail_;
            }

            // Bypass buffer for large remainders: read straight into caller.
            if (remaining >= BufferSize) {
                while (remaining > 0) {
                    std::size_t got = self().read_some(std::span<std::byte>{ out, remaining });
                    if (got == 0) STREAM_FAIL("IStream: unexpected EOF");
                    out += got;
                    remaining -= got;
                }
                return;
            }

            std::size_t got = refill();
            if (got == 0) STREAM_FAIL("IStream: unexpected EOF");
        }
    }

    // Generic typed read for any trivially-copyable T.  Pure bitwise copy.
    // Returns the Derived (CRTP self) for chaining with the most-derived type.
    template <class T> requires std::is_trivially_copyable_v<T>
    Derived& operator>>(T& v) {
        read_exact(std::as_writable_bytes(std::span<T, 1>(&v, 1)));
        return self();
    }
};

// ────────────────────────────────────────────────────────────────────────────
//  OStream: buffered byte-sink
// ────────────────────────────────────────────────────────────────────────────

template <std::size_t BufferSize, class Allocator, class Derived>
class OStream {
    static_assert(BufferSize > 0, "OStream BufferSize must be > 0");

    using ByteAlloc = typename std::allocator_traits<Allocator>
                          ::template rebind_alloc<std::byte>;
    using ByteAllocTraits = std::allocator_traits<ByteAlloc>;

    ByteAlloc alloc_;
    std::byte* buf_;
    std::byte* head_;  // next free slot

    Derived&       self()       noexcept { return static_cast<Derived&>(*this); }
    const Derived& self() const noexcept { return static_cast<const Derived&>(*this); }

    // Drain [buf_, head_) through Derived::write_some, repeating on partial writes.
    void drain() {
        std::byte* p = buf_;
        while (p < head_) {
            std::size_t n = static_cast<std::size_t>(head_ - p);
            std::size_t sent = self().write_some(std::span<const std::byte>{ p, n });
            if (sent == 0) STREAM_FAIL("OStream: write_some returned 0");
            p += sent;
        }
        head_ = buf_;
    }

   public:
    explicit OStream(const Allocator& alloc = Allocator{})
        : alloc_(alloc),
          buf_(ByteAllocTraits::allocate(alloc_, BufferSize)),
          head_(buf_) {}

    ~OStream() {
        static_assert(ByteSink<Derived>,
            "OStream<...> Derived must implement: "
            "std::size_t write_some(std::span<const std::byte>)");
        if (buf_) {
#if defined(__cpp_exceptions) && __cpp_exceptions
            // Best-effort flush; swallow errors in the destructor.
            try { drain(); } catch (...) {}
#else
            // Without exceptions, drain() will std::abort on a transport
            // failure; that's acceptable for the bare-metal target where
            // we'd have nowhere to surface the error anyway.
            drain();
#endif
            ByteAllocTraits::deallocate(alloc_, buf_, BufferSize);
        }
    }

    OStream(const OStream&) = delete;
    OStream& operator=(const OStream&) = delete;
    OStream(OStream&&) = delete;
    OStream& operator=(OStream&&) = delete;

    static constexpr std::size_t buffer_size() noexcept { return BufferSize; }
    std::size_t pending() const noexcept { return static_cast<std::size_t>(head_ - buf_); }

    // Write exactly src.size() bytes.  Buffers small writes; for writes that
    // would not fit, drains the buffer and pushes the payload through directly.
    void write_exact(std::span<const std::byte> src) {
        const std::byte* in = src.data();
        std::size_t remaining = src.size();

        // Fast path: everything fits in the remaining buffer.
        std::size_t free_space = static_cast<std::size_t>((buf_ + BufferSize) - head_);
        if (remaining <= free_space) {
            std::memcpy(head_, in, remaining);
            head_ += remaining;
            return;
        }

        // Otherwise drain whatever's already buffered first.
        drain();

        // Large payload: bypass the buffer and write straight from caller's memory.
        if (remaining >= BufferSize) {
            while (remaining > 0) {
                std::size_t sent = self().write_some(std::span<const std::byte>{ in, remaining });
                if (sent == 0) STREAM_FAIL("OStream: write_some returned 0");
                in += sent;
                remaining -= sent;
            }
            return;
        }

        // Small payload now fits in the freshly-drained buffer.
        std::memcpy(head_, in, remaining);
        head_ += remaining;
    }

    // Drain the buffer and (if Derived supports it) sync the underlying transport.
    void flush() {
        drain();
        if constexpr (HasSync<Derived>) {
            self().sync();
        }
    }

    template <class T> requires std::is_trivially_copyable_v<T>
    Derived& operator<<(const T& v) {
        write_exact(std::as_bytes(std::span<const T, 1>(&v, 1)));
        return self();
    }
};

// ────────────────────────────────────────────────────────────────────────────
//  IOStream: convenience combination via CRTP through both bases
// ────────────────────────────────────────────────────────────────────────────

template <std::size_t InBuf, std::size_t OutBuf, class Allocator, class Derived>
class IOStream
    : public IStream<InBuf,  Allocator, Derived>,
      public OStream<OutBuf, Allocator, Derived> {
   public:
    using in_base  = IStream<InBuf,  Allocator, Derived>;
    using out_base = OStream<OutBuf, Allocator, Derived>;

    explicit IOStream(const Allocator& alloc = Allocator{})
        : in_base(alloc), out_base(alloc) {}

    using in_base::operator>>;
    using out_base::operator<<;
};

// ────────────────────────────────────────────────────────────────────────────
//  DerivedFromIStream / DerivedFromOStream concepts.
//
//  Used to constrain the free-function tensor << / >> overloads so they only
//  participate in overload resolution for stream types (and don't accidentally
//  match arbitrary user types that happen to take a TensorBase&).
//
//  Detection relies on the [temp.deduct.call] derived-to-base rule: when the
//  parameter is a pointer to a class template specialization and the argument
//  is a pointer to a class derived from a unique specialization of that
//  template, deduction succeeds.  IOStream is matched implicitly since it
//  derives from both IStream and OStream.
// ────────────────────────────────────────────────────────────────────────────

namespace stream_detail {
    template <std::size_t N, class A, class D>
    std::true_type  is_istream_base(const volatile IStream<N, A, D>*);
    std::false_type is_istream_base(const volatile void*);

    template <std::size_t N, class A, class D>
    std::true_type  is_ostream_base(const volatile OStream<N, A, D>*);
    std::false_type is_ostream_base(const volatile void*);
}

template <class S>
concept DerivedFromIStream =
    decltype(stream_detail::is_istream_base(static_cast<S*>(nullptr)))::value;

template <class S>
concept DerivedFromOStream =
    decltype(stream_detail::is_ostream_base(static_cast<S*>(nullptr)))::value;

// ────────────────────────────────────────────────────────────────────────────
//  TensorBase << / >> overloads.
//  Raw bitwise transfer of static_size * sizeof(T) bytes through .data().
//  Constrained on T being trivially copyable so we don't accidentally claim
//  TensorBase<NonTrivial, ...>.
// ────────────────────────────────────────────────────────────────────────────

template <class S, class T, FixedExtent E>
    requires DerivedFromIStream<S> && std::is_trivially_copyable_v<T>
S& operator>>(S& s, TensorBase<T, E>& t) {
    constexpr std::size_t n = TensorBase<T, E>::static_size;
    s.read_exact(std::as_writable_bytes(std::span<T, n>(t.data(), n)));
    return s;
}

template <class S, class T, FixedExtent E>
    requires DerivedFromOStream<S> && std::is_trivially_copyable_v<T>
S& operator<<(S& s, const TensorBase<T, E>& t) {
    constexpr std::size_t n = TensorBase<T, E>::static_size;
    s.write_exact(std::as_bytes(std::span<const T, n>(t.data(), n)));
    return s;
}

#endif  // __STREAMS_H__
