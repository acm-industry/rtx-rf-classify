#ifndef __NETWORK_STREAM_H__
#define __NETWORK_STREAM_H__

#include <sys/socket.h>
#include <cerrno>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>

#include "network.h"
#include "streams.h"

/**
 * TCP-backed CRTP streams.
 *
 * Wraps an existing Connection (does not own it) and adapts ::send / ::recv
 * to the IStream / OStream byte-source / byte-sink contract.
 *
 * EOF semantics: read_some / write_some return 0 to signal a closed peer
 * (instead of throwing the way Connection::recv / Connection::send do); this
 * lets the IStream / OStream base classes surface end-of-stream cleanly.
 *
 * Real I/O errors (errno != EOF, e.g. ECONNRESET on writes, or a
 * truly unexpected condition) still throw a std::runtime_error.
 */

namespace tcp_stream_detail {

inline std::size_t recv_some(int sock, std::span<std::byte> dst) {
    if (dst.empty()) return 0;
    ssize_t n = ::recv(sock, dst.data(), dst.size(), 0);
    if (n == 0) return 0;            // peer closed cleanly
    if (n < 0) {
        if (errno == EINTR) return 0; // caller will retry via outer loop
        throw std::runtime_error("TcpIStream: recv failed");
    }
    return static_cast<std::size_t>(n);
}

inline std::size_t send_some(int sock, std::span<const std::byte> src) {
    if (src.empty()) return 0;
    ssize_t n = ::send(sock, src.data(), src.size(),
#ifdef MSG_NOSIGNAL
                       MSG_NOSIGNAL
#else
                       0
#endif
                       );
    if (n == 0) return 0;            // treat as "no progress" (rare)
    if (n < 0) {
        if (errno == EINTR) return 0;
        if (errno == EPIPE || errno == ECONNRESET) return 0; // peer gone
        throw std::runtime_error("TcpOStream: send failed");
    }
    return static_cast<std::size_t>(n);
}

}  // namespace tcp_stream_detail

// ────────────────────────────────────────────────────────────────────────────
//  TcpIStream / TcpOStream / TcpIOStream
// ────────────────────────────────────────────────────────────────────────────

template <std::size_t BufferSize = 4096,
          class Allocator        = std::allocator<std::byte>>
class TcpIStream
    : public IStream<BufferSize, Allocator,
                     TcpIStream<BufferSize, Allocator>> {
    Connection& conn_;

   public:
    explicit TcpIStream(Connection& conn, const Allocator& alloc = Allocator{})
        : IStream<BufferSize, Allocator, TcpIStream>(alloc), conn_(conn) {}

    std::size_t read_some(std::span<std::byte> dst) {
        return tcp_stream_detail::recv_some(conn_.native_handle(), dst);
    }
};

template <std::size_t BufferSize = 4096,
          class Allocator        = std::allocator<std::byte>>
class TcpOStream
    : public OStream<BufferSize, Allocator,
                     TcpOStream<BufferSize, Allocator>> {
    Connection& conn_;

   public:
    explicit TcpOStream(Connection& conn, const Allocator& alloc = Allocator{})
        : OStream<BufferSize, Allocator, TcpOStream>(alloc), conn_(conn) {}

    std::size_t write_some(std::span<const std::byte> src) {
        return tcp_stream_detail::send_some(conn_.native_handle(), src);
    }
};

template <std::size_t InBuf  = 4096,
          std::size_t OutBuf = 4096,
          class Allocator    = std::allocator<std::byte>>
class TcpIOStream
    : public IOStream<InBuf, OutBuf, Allocator,
                      TcpIOStream<InBuf, OutBuf, Allocator>> {
    Connection& conn_;

   public:
    explicit TcpIOStream(Connection& conn, const Allocator& alloc = Allocator{})
        : IOStream<InBuf, OutBuf, Allocator, TcpIOStream>(alloc), conn_(conn) {}

    std::size_t read_some(std::span<std::byte> dst) {
        return tcp_stream_detail::recv_some(conn_.native_handle(), dst);
    }

    std::size_t write_some(std::span<const std::byte> src) {
        return tcp_stream_detail::send_some(conn_.native_handle(), src);
    }
};

#endif  // __NETWORK_STREAM_H__
