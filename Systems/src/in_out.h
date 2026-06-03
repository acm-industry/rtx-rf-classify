#ifndef IN_OUT_H_
#define IN_OUT_H_

#include <cstddef>
#include <cstdint>
#include <span>

#include "rfio.h"

class RFIO {
    static constexpr size_t word_bytes = sizeof(uint32_t);

    uint32_t rx_word_ = 0;
    size_t rx_offset_ = word_bytes;

    static constexpr std::byte byte_from_word(uint32_t word, size_t offset) {
        return static_cast<std::byte>( static_cast<unsigned char>((word >> (8u * offset)) & 0xffu) );
    }

    static constexpr void set_word_byte(uint32_t& word, std::byte value, size_t offset) {
        word |= std::to_integer<uint32_t>(value) << (8u * offset);
    }

public:
    static constexpr uintptr_t rx_data_addr = RFIO_RX_DATA_ADDR;
    static constexpr uintptr_t tx_data_addr = RFIO_TX_DATA_ADDR;
    static constexpr uintptr_t status_addr = RFIO_STATUS_ADDR;

    RFIO() = default;

    RFIO(const RFIO&) = delete;
    RFIO& operator=(const RFIO&) = delete;

    RFIO(RFIO&&) noexcept = default;
    RFIO& operator=(RFIO&&) noexcept = default;

    void send(std::span<const std::byte> buffer) {
        const std::byte* ptr = buffer.data();
        size_t remaining = buffer.size();

        while (remaining >= word_bytes) {
            uint32_t word = 0;
            for (size_t i = 0; i < word_bytes; ++i) set_word_byte(word, ptr[i], i);

            rfio_write_u32(word);
            ptr += word_bytes;
            remaining -= word_bytes;
        }

        if (remaining > 0) {
            // RFIO TX is word-wide; trailing bytes are zero-padded
            uint32_t word = 0;
            for (size_t i = 0; i < remaining; ++i) set_word_byte(word, ptr[i], i);
            rfio_write_u32(word);
        }
    }

    void recv(std::span<std::byte> buffer) {
        std::byte* ptr = buffer.data();
        size_t remaining = buffer.size();

        while (remaining > 0) {
            if (rx_offset_ == word_bytes) {
                rx_word_ = rfio_read_u32();
                rx_offset_ = 0;
            }

            *ptr = byte_from_word(rx_word_, rx_offset_);

            ++ptr;
            ++rx_offset_;
            --remaining;
        }
    }

    uint32_t status() const {
        return rfio_status();
    }

    bool connected() const {
        return (status() & RFIO_STATUS_CONNECTED) != 0;
    }
};

#endif  
