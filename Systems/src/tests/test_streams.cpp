// Unit tests for streams.h and network_stream.h.
//
// Coverage:
//   1. MemoryStream (in-memory derived) round-trip of trivially-copyable types,
//      structs, and a TensorBase view.
//   2. Buffer-boundary cases: refill across small read_some chunks, drain across
//      small write_some chunks, payloads larger than the BufferSize that bypass
//      the buffer entirely.
//   3. EOF surfacing (read_exact throws when the source is exhausted).
//   4. TCP loopback of the exact MVP wire format (batch_len + N float tensors)
//      plus the matching argmax bytes coming back.

#include <arpa/inet.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../network.h"
#include "../network_stream.h"
#include "../streams.h"
#include "../tensor.h"

namespace {

int total = 0;
int failed = 0;

void check(const char* name, bool ok) {
    ++total;
    if (!ok) {
        ++failed;
        std::cout << "  [FAIL] " << name << "\n";
    } else {
        std::cout << "  [ OK ] " << name << "\n";
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  MemoryStream: in-memory IOStream Derived used both as a unit-test mock and
//  as a reference implementation for what a bare-metal "static-buffer" stream
//  would look like.  Reads draw from an in-vector cursor; writes append to a
//  separate out-vector.
// ────────────────────────────────────────────────────────────────────────────

template <std::size_t InBuf  = 16,
          std::size_t OutBuf = 16,
          class Allocator    = std::allocator<std::byte>>
class MemoryStream
    : public IOStream<InBuf, OutBuf, Allocator,
                      MemoryStream<InBuf, OutBuf, Allocator>> {
   public:
    std::vector<std::byte> in_data;
    std::vector<std::byte> out_data;
    std::size_t in_cursor = 0;
    // Cap how many bytes we hand back per read_some / accept per write_some
    // to exercise the buffer's loop / refill / drain code paths.
    std::size_t read_chunk  = 0;  // 0 == no cap
    std::size_t write_chunk = 0;

    using base_t = IOStream<InBuf, OutBuf, Allocator, MemoryStream>;
    using base_t::base_t;

    void load(std::span<const std::byte> bytes) {
        in_data.assign(bytes.begin(), bytes.end());
        in_cursor = 0;
    }

    std::size_t read_some(std::span<std::byte> dst) {
        std::size_t left = in_data.size() - in_cursor;
        if (left == 0) return 0;
        std::size_t n = dst.size();
        if (n > left) n = left;
        if (read_chunk && n > read_chunk) n = read_chunk;
        std::memcpy(dst.data(), in_data.data() + in_cursor, n);
        in_cursor += n;
        return n;
    }

    std::size_t write_some(std::span<const std::byte> src) {
        std::size_t n = src.size();
        if (write_chunk && n > write_chunk) n = write_chunk;
        out_data.insert(out_data.end(), src.data(), src.data() + n);
        return n;
    }
};

// ────────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────────

void test_pod_round_trip() {
    std::cout << "Test: trivially-copyable round trip\n";

    MemoryStream<> s;
    const std::uint32_t a = 0xDEADBEEF;
    const float        b = 1.5f;
    const std::int8_t  c = -3;
    s << a << b << c;
    s.flush();

    s.load(s.out_data);
    std::uint32_t a2 = 0; float b2 = 0; std::int8_t c2 = 0;
    s >> a2 >> b2 >> c2;

    check("uint32 round trip", a2 == a);
    check("float round trip",  b2 == b);
    check("int8 round trip",   c2 == c);
}

void test_tensor_round_trip() {
    std::cout << "Test: TensorBase round trip via operator<< / operator>>\n";

    constexpr std::size_t N = 3 * 128;
    std::array<float, N> in_buf, out_buf{};

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : in_buf) v = dist(rng);

    using View = TensorBase<float, std::extents<std::size_t, 3, 128>>;
    View in_t  { std::span<float, N>(in_buf.data(),  N) };
    View out_t { std::span<float, N>(out_buf.data(), N) };

    MemoryStream<> s;
    s << in_t;
    s.flush();
    s.load(s.out_data);
    s >> out_t;

    check("tensor wire byte count == 3*128*sizeof(float)",
          s.out_data.size() == N * sizeof(float));
    check("tensor contents match",
          std::memcmp(in_buf.data(), out_buf.data(), N * sizeof(float)) == 0);
}

void test_partial_read_write() {
    std::cout << "Test: partial read_some / write_some chunks\n";

    // Tiny in/out buffers + 1-byte chunks at the transport => stream must
    // refill / drain many times to satisfy a multi-byte read_exact / write_exact.
    MemoryStream<8, 8> s;
    s.read_chunk  = 1;
    s.write_chunk = 1;

    const std::array<std::uint64_t, 5> src = {1, 2, 3, 4, 5};
    for (auto v : src) s << v;
    s.flush();

    check("write produced full payload",
          s.out_data.size() == src.size() * sizeof(std::uint64_t));

    s.load(s.out_data);
    std::array<std::uint64_t, 5> dst{};
    for (auto& v : dst) s >> v;

    bool ok = true;
    for (std::size_t i = 0; i < src.size(); ++i) ok = ok && (dst[i] == src[i]);
    check("values survived 1-byte chunked transport", ok);
}

void test_large_payload_bypass() {
    std::cout << "Test: payload larger than buffer bypasses buffer\n";

    // Buffer is 8 bytes; payload is 4096 bytes -> must hit the bypass path.
    constexpr std::size_t N = 4096;
    std::vector<std::byte> payload(N);
    for (std::size_t i = 0; i < N; ++i) payload[i] = static_cast<std::byte>(i & 0xFF);

    MemoryStream<8, 8> s;
    s.write_exact(payload);
    s.flush();
    check("large write produced full payload", s.out_data.size() == N);
    check("large write byte content matches",
          std::memcmp(payload.data(), s.out_data.data(), N) == 0);

    s.load(s.out_data);
    std::vector<std::byte> readback(N);
    s.read_exact(readback);
    check("large read content matches",
          std::memcmp(payload.data(), readback.data(), N) == 0);
}

void test_eof_throws() {
    std::cout << "Test: read past EOF throws\n";

    MemoryStream<> s;
    const std::uint32_t v = 0x1234;
    s << v;
    s.flush();
    s.load(s.out_data);

    std::uint32_t a = 0, b = 0;
    bool threw = false;
    try {
        s >> a >> b;  // second read should hit EOF
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check("EOF surfaced as runtime_error", threw);
    check("first read still succeeded", a == v);
}

// ────────────────────────────────────────────────────────────────────────────
//  TCP loopback exercising the exact MVP wire format.
//  Server thread runs a tiny server that uses TcpIOStream to read
//  (batch_len + N tensors of 3*128 floats), echoes back N argmax bytes
//  (computed as the index of the max element in the first row).
//  Client uses TcpIOStream to drive the same wire format and verify.
// ────────────────────────────────────────────────────────────────────────────

uint16_t pick_port() {
    // Pseudo-random port in the ephemeral range; SO_REUSEADDR is set in
    // Server so back-to-back runs work even on TIME_WAIT.
    static std::atomic<uint16_t> seed{40000};
    return seed.fetch_add(7) | 1;
}

void test_tcp_loopback() {
    std::cout << "Test: TCP loopback with MVP wire format\n";

    constexpr std::size_t TENSOR_FLOATS = 3 * 128;
    constexpr std::uint32_t batch_len   = 4;

    std::vector<std::array<float, TENSOR_FLOATS>> inputs(batch_len);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& t : inputs) for (auto& v : t) v = dist(rng);

    std::vector<std::uint8_t> expected(batch_len);
    for (std::uint32_t i = 0; i < batch_len; ++i) {
        std::size_t best = 0;
        for (std::size_t j = 1; j < 11 && j < TENSOR_FLOATS; ++j) {
            if (inputs[i][j] > inputs[i][best]) best = j;
        }
        expected[i] = static_cast<std::uint8_t>(best);
    }

    uint16_t port = pick_port();
    std::atomic<bool> server_ready{false};
    std::atomic<bool> server_ok{false};

    std::thread server_thread([&] {
        try {
            Server srv(port);
            server_ready.store(true);
            Connection conn = srv.accept();
            TcpIOStream<> io(conn);

            std::uint32_t bl_net = 0;
            io >> bl_net;
            std::uint32_t bl = ntohl(bl_net);

            using View = TensorBase<float, std::extents<std::size_t, 3, 128>>;
            std::array<float, TENSOR_FLOATS> scratch{};
            View tv { std::span<float, TENSOR_FLOATS>(scratch.data(), TENSOR_FLOATS) };

            for (std::uint32_t i = 0; i < bl; ++i) {
                io >> tv;
                std::size_t best = 0;
                for (std::size_t j = 1; j < 11 && j < TENSOR_FLOATS; ++j) {
                    if (scratch[j] > scratch[best]) best = j;
                }
                std::uint8_t ans = static_cast<std::uint8_t>(best);
                io << ans;
            }
            io.flush();
            server_ok.store(true);
        } catch (const std::exception& e) {
            std::cout << "  [server thread] exception: " << e.what() << "\n";
        }
    });

    // Wait for the server to bind / listen before connecting.
    for (int i = 0; i < 200 && !server_ready.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    check("server thread became ready", server_ready.load());

    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    int rc = ::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    check("client connected to server", rc == 0);

    {
        Connection client(sock);
        TcpIOStream<> io(client);

        std::uint32_t bl_net = htonl(batch_len);
        io << bl_net;
        using View = TensorBase<float, std::extents<std::size_t, 3, 128>>;
        for (auto& t : inputs) {
            View tv { std::span<float, TENSOR_FLOATS>(t.data(), TENSOR_FLOATS) };
            io << tv;
        }
        io.flush();

        std::vector<std::uint8_t> got(batch_len);
        for (auto& g : got) io >> g;

        bool match = std::equal(got.begin(), got.end(), expected.begin());
        check("argmax bytes round-tripped through MVP wire format", match);
    }  // client Connection closes here -> EOF on server

    server_thread.join();
    check("server thread completed cleanly", server_ok.load());
}

}  // namespace

int main() {
    std::cout << "================================================\n";
    std::cout << "  streams.h / network_stream.h tests\n";
    std::cout << "================================================\n";

    test_pod_round_trip();
    test_tensor_round_trip();
    test_partial_read_write();
    test_large_payload_bypass();
    test_eof_throws();
    test_tcp_loopback();

    std::cout << "------------------------------------------------\n";
    std::cout << "  passed: " << (total - failed) << " / " << total << "\n";
    if (failed > 0) std::cout << "  FAILED: " << failed << "\n";
    return failed == 0 ? 0 : 1;
}
