/**
 * Native TCP entry point for the classification MVP.
 *
 * The actual inference pipeline lives in inference.h and is shared with the
 * bare-metal Ara entry point (main_baremetal.cpp).  This file only owns the
 * Berkeley-sockets server loop + per-connection TcpIOStream construction.
 */

#include <iostream>

#include "memorybuffer.h"
#include "network.h"
#include "network_stream.h"
#include "inference.h"

int main() {

  static constexpr size_t INPUT_OUTPUT_ALLOC_BYTES = ( 3 * 128 + 128 * 32 ) * sizeof(float) + 256;
  MemoryBuffer buf( INPUT_OUTPUT_ALLOC_BYTES ); // Alloc enough for input/output tensor plus some wiggle room
  auto alloc = buf.get_allocator<float, 32>();

  f32Tensor<3, 128>  input(alloc);
  f32Tensor<128, 32> output(alloc);

  Server server(8080);

  std::cout << "Listening on port 8080.\n";

  for (;;) {
    Connection conn = server.accept();

    std::cout << "Recieved connection. Running compute...\n";

    try {
      TcpIOStream<> io(conn);
      serve_inference(io, input, output);
      std::cout << "Batch sent.\n";
    } catch ( const std::runtime_error& err ) {
      std::cout << "Connection closed. Killing compute.\n";
    }

  }

  return 0;
}
