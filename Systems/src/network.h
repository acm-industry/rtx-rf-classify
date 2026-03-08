#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <span>
#include <stdexcept>
#include <cstring>

/**
 * MVP-Specific network protocols
 * Very basic, minimum product for networking to work
 * Later to be replaced based on FPGA or simulatr I/O
 * For right now, just a network sock
 */

class Connection {
    int sock_;

public:
    explicit Connection(int sock) : sock_(sock) {}

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    Connection(Connection&& other) noexcept : sock_(other.sock_) {
        other.sock_ = -1;
    }

    ~Connection() {
        if (sock_ >= 0) ::close(sock_);
    }

    void send(std::span<const std::byte> buffer) {
        const std::byte* ptr = buffer.data();
        size_t remaining = buffer.size();

        while (remaining > 0) {
            ssize_t sent = ::send(sock_, ptr, remaining, 0);
            if (sent <= 0) throw std::runtime_error("send failed");
            ptr += sent;
            remaining -= sent;
        }
    }

    void recv(std::span<std::byte> buffer) {
        std::byte* ptr = buffer.data();
        size_t remaining = buffer.size();

        while (remaining > 0) {
            ssize_t recvd = ::recv(sock_, ptr, remaining, 0);
            if (recvd <= 0) throw std::runtime_error("recv failed");
            ptr += recvd;
            remaining -= recvd;
        }
    }

    int native_handle() const { return sock_; }
};

class Server {
    int server_sock_;

public:
    explicit Server(uint16_t port) {
        server_sock_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server_sock_ < 0) throw std::runtime_error("socket failed");

        int opt = 1;
        ::setsockopt(server_sock_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton( AF_INET, "127.0.0.1", &addr.sin_addr );

        if (::bind(server_sock_, (sockaddr*)&addr, sizeof(addr)) < 0) throw std::runtime_error("bind failed");

        if (::listen(server_sock_, 8) < 0) throw std::runtime_error("listen failed");
    }

    ~Server() {
        if (server_sock_ >= 0) ::close(server_sock_);
    }

    Connection accept() {
        int client = ::accept(server_sock_, nullptr, nullptr);
        if (client < 0) throw std::runtime_error("accept failed");
        return Connection(client);
    }
};




#endif
