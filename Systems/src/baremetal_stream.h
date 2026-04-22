#ifndef __BAREMETAL_STREAM_H__
#define __BAREMETAL_STREAM_H__

#include <cstddef>
#include <cstdint>
#include <span>

#include "streams.h"

/**
 * Bare-metal CRTP stream backends for Ara / Verilator.
 *
 * No heap, no exceptions, no Berkeley sockets.  Input is consumed from a
 * fixed buffer of bytes that the linker baked into .rodata via .incbin
 * (see baremetal_input.S).  Output goes through Ara's printf-over-HTIF so
 * the host-side verify_rtl_simulation.sh can grep predictions out of
 * build/rtl_sim_output.txt.
 *
 * The IStream / OStream base classes ask their Allocator for BufferSize
 * bytes on construction.  Newlib's malloc would work in principle, but we
 * keep things explicit (and avoid pulling _sbrk / heap setup) by providing
 * a tiny per-instance static-pool allocator.  Each stream owns a private
 * static byte arena sized to BufferSize.
 *
 * On the output side we emit each byte as a two-character lowercase hex
 * digit between `[ARGMAX-START]` and `[ARGMAX-END]` sentinels so the host
 * script can extract predictions with a single `sed`/`awk`.
 *
 * Wire format (matches Phase 1 MVP):
 *   in : uint32_t batch_len_net (network byte order, big-endian),
 *        then batch_len * (3 * 128) floats (host byte order, raw bytes).
 *   out: batch_len * uint8_t argmax in [0, 11), one hex pair per byte.
 */

// printf.h is provided by the Ara runtime; on a host build we accept any
// printf available through <cstdio>.  The macro indirection lets us swap
// implementations without changing the call sites.
#if defined(BAREMETAL_USE_ARA_PRINTF)
extern "C" int printf(const char* fmt, ...);
#  define BM_PRINTF ::printf
#else
#  include <cstdio>
#  define BM_PRINTF std::printf
#endif

// ────────────────────────────────────────────────────────────────────────────
// StaticPoolAllocator: hand back a chunk of a fixed-size static buffer.
//
// Instantiated with a unique tag per stream so each stream's buffer is
// distinct (avoids two streams accidentally sharing one byte arena).  Bump
// allocator only; deallocate is a no-op and we don't reuse memory.
// ────────────────────────────────────────────────────────────────────────────

namespace baremetal_stream_detail {

template <class Tag, std::size_t Capacity>
class StaticPoolAllocator {
   public:
    using value_type = std::byte;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap            = std::true_type;
    using is_always_equal                        = std::true_type;

    template <class U>
    struct rebind { using other = StaticPoolAllocator<Tag, Capacity>; };

    StaticPoolAllocator() = default;
    template <class U>
    StaticPoolAllocator(const StaticPoolAllocator<Tag, Capacity>&) noexcept {}

    std::byte* allocate(std::size_t n) {
        if (n > Capacity) {
#if defined(__cpp_exceptions) && __cpp_exceptions
            throw std::bad_alloc();
#else
            std::abort();
#endif
        }
        return pool_;
    }

    void deallocate(std::byte*, std::size_t) noexcept {}

    bool operator==(const StaticPoolAllocator&) const noexcept { return true; }
    bool operator!=(const StaticPoolAllocator&) const noexcept { return false; }

   private:
    static inline std::byte pool_[Capacity]{};
};

// Tag types so each stream class gets its own pool instantiation.
struct InTag {};
struct OutTag {};

}  // namespace baremetal_stream_detail

// ────────────────────────────────────────────────────────────────────────────
// StaticBufferIStream: read_some over a const std::byte* / size_t cursor.
//
// `start` and `end` are the bounds of the baked input fixture.  EOF is
// signalled by returning 0 from read_some, which IStream::read_exact then
// surfaces via STREAM_FAIL (abort on bare-metal, throw natively).
// ────────────────────────────────────────────────────────────────────────────

template <std::size_t BufferSize = 1024,
          class Allocator = baremetal_stream_detail::StaticPoolAllocator<
              baremetal_stream_detail::InTag, BufferSize>>
class StaticBufferIStream
    : public IStream<BufferSize, Allocator,
                     StaticBufferIStream<BufferSize, Allocator>> {
    const std::byte* cursor_;
    const std::byte* end_;

   public:
    StaticBufferIStream(const std::byte* start, const std::byte* end,
                        const Allocator& alloc = Allocator{})
        : IStream<BufferSize, Allocator, StaticBufferIStream>(alloc),
          cursor_(start),
          end_(end) {}

    // Convenience: take the .incbin _start / _end pair as unsigned char[]
    // (the form maweights.cpp / baremetal_input.S use for ELF visibility).
    StaticBufferIStream(const unsigned char* start, const unsigned char* end,
                        const Allocator& alloc = Allocator{})
        : StaticBufferIStream(reinterpret_cast<const std::byte*>(start),
                              reinterpret_cast<const std::byte*>(end),
                              alloc) {}

    std::size_t read_some(std::span<std::byte> dst) {
        if (dst.empty() || cursor_ >= end_) return 0;
        std::size_t avail = static_cast<std::size_t>(end_ - cursor_);
        std::size_t n     = dst.size() < avail ? dst.size() : avail;
        for (std::size_t i = 0; i < n; ++i) dst[i] = cursor_[i];
        cursor_ += n;
        return n;
    }

    std::size_t bytes_remaining() const noexcept {
        return cursor_ < end_ ? static_cast<std::size_t>(end_ - cursor_) : 0;
    }
};

// ────────────────────────────────────────────────────────────────────────────
// HtifOStream: write_some emits hex bytes through printf-over-HTIF.
//
// Format on the wire (host stdout / rtl_sim_output.txt):
//   [ARGMAX-START]<hh><hh>...<hh>[ARGMAX-END]
// where each <hh> is one byte of payload printed as %02x.  flush() emits the
// closing sentinel; the opening one is emitted lazily on the first byte so
// zero-length runs leave no trace.
// ────────────────────────────────────────────────────────────────────────────

template <std::size_t BufferSize = 256,
          class Allocator = baremetal_stream_detail::StaticPoolAllocator<
              baremetal_stream_detail::OutTag, BufferSize>>
class HtifOStream
    : public OStream<BufferSize, Allocator,
                     HtifOStream<BufferSize, Allocator>> {
    bool started_ = false;

    void start_if_needed() {
        if (!started_) {
            BM_PRINTF("[ARGMAX-START]");
            started_ = true;
        }
    }

   public:
    explicit HtifOStream(const Allocator& alloc = Allocator{})
        : OStream<BufferSize, Allocator, HtifOStream>(alloc) {}

    std::size_t write_some(std::span<const std::byte> src) {
        if (src.empty()) return 0;
        start_if_needed();
        for (std::byte b : src) {
            BM_PRINTF("%02x", static_cast<unsigned>(std::to_integer<uint8_t>(b)));
        }
        return src.size();
    }

    // Called by OStream::flush() after the buffer drains.  Closes the
    // sentinel and inserts a newline so the host script can find it.
    void sync() {
        if (started_) {
            BM_PRINTF("[ARGMAX-END]\n");
            started_ = false;
        }
    }
};

// ────────────────────────────────────────────────────────────────────────────
// BareIOStream: thin convenience composition of the two halves.
//
// IOStream's CRTP self-type has to expose both read_some and write_some, so
// we forward into the appropriate transport here rather than duplicating the
// state in the bases.
// ────────────────────────────────────────────────────────────────────────────

template <std::size_t InBuf  = 1024,
          std::size_t OutBuf = 256,
          class InAllocator  = baremetal_stream_detail::StaticPoolAllocator<
              baremetal_stream_detail::InTag, InBuf>,
          class OutAllocator = baremetal_stream_detail::StaticPoolAllocator<
              baremetal_stream_detail::OutTag, OutBuf>>
class BareIOStream
    : public IOStream<InBuf, OutBuf, InAllocator,
                      BareIOStream<InBuf, OutBuf, InAllocator, OutAllocator>> {
    const std::byte* cursor_;
    const std::byte* end_;
    bool             started_ = false;

    void start_if_needed() {
        if (!started_) {
            BM_PRINTF("[ARGMAX-START]");
            started_ = true;
        }
    }

   public:
    using base = IOStream<InBuf, OutBuf, InAllocator, BareIOStream>;

    BareIOStream(const std::byte* start, const std::byte* end,
                 const InAllocator&  in_alloc  = InAllocator{},
                 const OutAllocator& out_alloc = OutAllocator{})
        : base(in_alloc),
          cursor_(start),
          end_(end) {
        // out_alloc is unused here because IOStream<> inherits OStream with
        // the input allocator type for symmetry; HtifOStream's static pool
        // is keyed on a separate Tag so the two pools don't collide.
        (void)out_alloc;
    }

    BareIOStream(const unsigned char* start, const unsigned char* end,
                 const InAllocator&  in_alloc  = InAllocator{},
                 const OutAllocator& out_alloc = OutAllocator{})
        : BareIOStream(reinterpret_cast<const std::byte*>(start),
                       reinterpret_cast<const std::byte*>(end),
                       in_alloc, out_alloc) {}

    std::size_t read_some(std::span<std::byte> dst) {
        if (dst.empty() || cursor_ >= end_) return 0;
        std::size_t avail = static_cast<std::size_t>(end_ - cursor_);
        std::size_t n     = dst.size() < avail ? dst.size() : avail;
        for (std::size_t i = 0; i < n; ++i) dst[i] = cursor_[i];
        cursor_ += n;
        return n;
    }

    std::size_t write_some(std::span<const std::byte> src) {
        if (src.empty()) return 0;
        start_if_needed();
        for (std::byte b : src) {
            BM_PRINTF("%02x", static_cast<unsigned>(std::to_integer<uint8_t>(b)));
        }
        return src.size();
    }

    void sync() {
        if (started_) {
            BM_PRINTF("[ARGMAX-END]\n");
            started_ = false;
        }
    }

    std::size_t bytes_remaining() const noexcept {
        return cursor_ < end_ ? static_cast<std::size_t>(end_ - cursor_) : 0;
    }
};

#endif  // __BAREMETAL_STREAM_H__
