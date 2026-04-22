#ifndef __INFERENCE_H__
#define __INFERENCE_H__

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "tensor.h"
#include "convolve.h"
#include "batchnorm.h"
#include "ExprSystem/Broadcast.h"
#include "ExprSystem/Expression.h"
#include "ExprSystem/Scalar.h"
#include "ExprSystem/ExprFunctions.h"
#include "memorybuffer.h"
#include "maxpool.h"
#include "avgpool.h"
#include "maweights.cpp"
#include "streams.h"

/**
 * Backend-agnostic inference pipeline.
 *
 * Pulled out of main.cpp so the native (TCP) and bare-metal (Ara) entry
 * points can share one implementation.  This header avoids any platform-
 * specific dependency: no <iostream>, no <arpa/inet.h>, no networking,
 * no <ranges>.  All I/O happens through the templated `IO&` parameter,
 * which must satisfy the IStream + OStream CRTP contract from streams.h.
 *
 * Wire format (matches Phase 1 MVP):
 *   in : uint32_t batch_len_net (network byte order, big-endian),
 *        then batch_len * (3 * 128) floats (host byte order, raw bytes).
 *   out: batch_len * uint8_t argmax in [0, 11).
 */

template <size_t... idxs> requires ( sizeof...(idxs) > 0 )
using f32Tensor = DynTensor<float, std::extents<size_t, idxs...>, MemoryBuffer::Allocator<float, 32>>;

template <size_t... idxs> requires ( sizeof...(idxs) > 0 )
using f32TensorView = TensorBase<float, std::extents<size_t, idxs...>>;


// ────────────────────────────────────────────────────────────────────────────
// Portable network-to-host 32-bit byte swap.
//
// Avoids the <arpa/inet.h> dependency so this header compiles on bare-metal
// newlib targets.  Mirrors the behaviour of POSIX ntohl: on little-endian
// hosts (x86-64, RISC-V Ara, Apple silicon) it byteswaps; on big-endian
// hosts (rare) it's the identity.
// ────────────────────────────────────────────────────────────────────────────
static inline uint32_t inference_ntohl(uint32_t x) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return x;
#else
    return __builtin_bswap32(x);
#endif
}


template <size_t padding, FixedExtent E1, FixedExtent E2, FixedExtent E3, FixedExtent E4>
    requires ( E1::rank() == 2 && E2::rank() == 3 && E3::rank() == 1 && E4::rank() == 2 )
void BroadcastConvolve(
  const TensorBase<float, E1>& input,
  const TensorBase<float, E2>& weights,
  const TensorBase<float, E3>& bias,
  TensorBase<float, E4>& out
) {

  std::array<float, E4::static_extent(1)> membuf;
  f32TensorView<E4::static_extent(1)> partial{ std::span(membuf) };

  for (size_t oc = 0; oc < E2::static_extent(0); ++oc ) {
    auto out_oc = out[oc];

    float b = bias[oc];
    for (size_t i = 0; i < E4::static_extent(1); ++i) out_oc[i] = b;

    for (size_t ic = 0; ic < E2::static_extent(1); ++ic) {
      auto x_ic = input[ic];
      auto w_oc_ic = weights[oc, ic];

      Conv1DInPlace<padding>(x_ic, w_oc_ic, partial);

      for (size_t i = 0; i < E4::static_extent(1); ++i) out_oc[i] += partial[i];
    }

  }

}

inline void FeatureExtractor( const f32TensorView<3, 128>& input, f32TensorView<128, 32>& out ) {
  /** -----------------------------------------------------
   *  Layer One, Conv1D -> BatchNorm1D -> ReLU -> MaxPool1D
   *  -----------------------------------------------------
   */

  // Memory buffer to reuse the same stack memory moving from tensor to tensor.
  // Largest size we have is 128 * 64 on layer two. So we need max two temporaries of size 128 * 64. Plus a bit of wiggle room in case (+ 256).
  std::array<std::byte, 128 * 64 * sizeof(float) + 256> memory1;
  MemoryBuffer buf1( std::span{memory1} );
  auto allocator1 = buf1.get_allocator<float, 32>();

  std::array<std::byte, 128 * 64 * sizeof(float) + 256> memory2;
  MemoryBuffer buf2( std::span{memory2} );
  auto allocator2 = buf2.get_allocator<float, 32>();

  f32Tensor<64, 128> conv_out_layer1(allocator1);
  f32TensorView<64, 3, 5> conv_weights_layer1{ std::span<float, 64 * 3 * 5>(weights_conv_1d_l1, 64 * 3 * 5) };

  f32TensorView<64> conv_bias_weights_layer1{ std::span<float, 64>(weights_conv_1d_bias_l1, 64) };

  BroadcastConvolve<2>( input, conv_weights_layer1, conv_bias_weights_layer1, conv_out_layer1 );

  f32TensorView<4, 64> bn_weights_layer1{ std::span<float, 4 * 64>( weights_bn_1d_l1, 4 * 64 ) };

  BatchNorm1DInPlace( conv_out_layer1, bn_weights_layer1, conv_out_layer1 );

  in_place_eval( relu(conv_out_layer1.as_view()), conv_out_layer1 );

  f32Tensor<64, 64> mp_out_layer1(allocator2);

  for (size_t i = 0; i < 64; ++i) {
    auto outvec = mp_out_layer1[i];
    MaxPool1DInPlace<2, 2>( conv_out_layer1[i] , outvec );
  }

  // Notice that, from now on, conv_out_layer1 goes unused. We can just reset the memory pool and reuse the same memory.
  buf1.reset();

  /** -----------------------------------------------------
   *  Layer Two, Conv1D -> BatchNorm1D -> ReLU -> MaxPool1D
   *  -----------------------------------------------------
   */

  f32TensorView<128, 64, 5> conv_weights_layer2( std::span<float, 128 * 64 * 5>( weights_conv_1d_l2, 128 * 64 * 5 ) );
  f32TensorView<128> conv_bias_weights_layer2{ std::span<float, 128>(weights_conv_1d_bias_l2, 128) };
  f32Tensor<128, 64> conv_out_layer2( allocator1 );

  BroadcastConvolve<2>( mp_out_layer1, conv_weights_layer2, conv_bias_weights_layer2, conv_out_layer2 );

  f32TensorView<4, 128> bn_weights_layer2( std::span<float, 4 * 128>( weights_bn_1d_l2, 4 * 128 ) );

  BatchNorm1DInPlace( conv_out_layer2, bn_weights_layer2, conv_out_layer2 );

  in_place_eval( relu( conv_out_layer2.as_view() ), conv_out_layer2 );

  buf2.reset();

  f32Tensor<128, 32> mp_out_layer2(allocator2);

  for (size_t i = 0; i < 128; ++i) {
    auto outvec = mp_out_layer2[i];

    MaxPool1DInPlace<2, 2>( conv_out_layer2[i], outvec );
  }

  // at this point, co_layer2 is using alloc1, mpout_layer2 is using alloc2
  buf1.reset();

  /** ------------------------------------------
   *  Layer Three, Conv1D -> BatchNorm1D -> ReLU
   *  ------------------------------------------
   */

  f32TensorView<128, 128, 3> conv_weights_layer3( std::span<float, 128 * 128 * 3>( weights_conv_1d_l3, 128 * 128 * 3 ) );
  f32TensorView<128> conv_bias_weights_layer3{ std::span<float, 128>(weights_conv_1d_bias_l3, 128) };
  f32Tensor<128, 32> conv_out_layer3( allocator1 );

  BroadcastConvolve<1>( mp_out_layer2, conv_weights_layer3, conv_bias_weights_layer3, conv_out_layer3 );

  f32TensorView<4, 128> bn_weights_layer3( std::span<float, 4 * 128>( weights_bn_1d_l3, 4 * 128 ) );

  BatchNorm1DInPlace( conv_out_layer3, bn_weights_layer3, conv_out_layer3 );

  in_place_eval( relu( conv_out_layer3.as_view() ), conv_out_layer3 );

  /** -----------------------------------------
   *  Layer Four, Conv1D -> BatchNorm1D -> ReLU
   *  -----------------------------------------
   */

  buf2.reset();

  f32TensorView<128, 128, 3> conv_weights_layer4( std::span<float, 128 * 128 * 3>( weights_conv_1d_l4, 128 * 128 * 3 ) );
  f32TensorView<128> conv_bias_weights_layer4{ std::span<float, 128>(weights_conv_1d_bias_l4, 128) };
  f32Tensor<128, 32> conv_out_layer4(allocator2);

  BroadcastConvolve<1>( conv_out_layer3, conv_weights_layer4, conv_bias_weights_layer4, conv_out_layer4 );

  f32TensorView<4, 128> bn_weights_layer4( std::span<float, 4 * 128>( weights_bn_1d_l4, 4 * 128 ) );

  BatchNorm1DInPlace( conv_out_layer4, bn_weights_layer4, conv_out_layer4 );

  in_place_eval( relu( conv_out_layer4.as_view() ), out );

}

inline void ClassifierHead(const f32TensorView<128, 32>& input, f32TensorView<11>& out ) {

  // Same idea as FeatureExtractor; preallocate the stack memory we will use for temporaries.
  std::array<std::byte, 128 * sizeof(float) + 256> memory1;
  MemoryBuffer buf1( std::span{memory1} );
  auto allocator1 = buf1.get_allocator<float, 32>();

  std::array<std::byte, 128 * sizeof(float) + 256> memory2;
  MemoryBuffer buf2( std::span{memory2} );
  auto allocator2 = buf2.get_allocator<float, 32>();

  f32Tensor<128, 1> avgpool_out( allocator1 );

  for (size_t i = 0; i < 128; ++i) {
    auto input_slice = input[i];
    auto output_slice = avgpool_out[i];
    AdaptiveAvgPool1DInPlace<1>( input_slice, output_slice );
  }

  f32TensorView<128> new_view{ std::span< float, 128 >( avgpool_out.data(), 128 ) };
  f32TensorView<128, 128> linear_mat_weights_1{ std::span<float, 128 * 128>( linear_mat_1, 128 * 128 ) };
  f32Tensor<128> linear_out_1( allocator2 );

  blas::gemv(linear_mat_weights_1, new_view, linear_out_1);

  f32TensorView<128> linear_add_weights_1{ std::span<float, 128>( linear_add_1, 128 ) };

  in_place_eval( relu( linear_out_1.as_view() + linear_add_weights_1), linear_out_1 );

  f32TensorView<11, 128> linear_mat_weights_2{ std::span<float, 11 * 128>( linear_mat_2, 11 * 128 ) };

  blas::gemv( linear_mat_weights_2, linear_out_1, out );

  f32TensorView<11> final_add{ std::span<float, 11>( linear_add_2, 11 ) };

  in_place_eval( out + final_add, out );
}

/**
 * Backend-agnostic per-batch MVP loop.
 *
 * Templated on the stream type so we can plug in TcpIOStream natively and a
 * bare-metal IOStream (HTIF + baked-in input) on Ara without touching this
 * function.  The argmax loop is hand-rolled (instead of std::ranges::
 * max_element) so this header stays cheap to include on newlib targets that
 * may not ship the <ranges> header at all.
 */
template <class IO>
void serve_inference(IO& io,
                     f32Tensor<3, 128>& input,
                     f32Tensor<128, 32>& output) {
  uint32_t batch_len_net;
  io >> batch_len_net;
  const uint32_t batch_len = inference_ntohl( batch_len_net );

  if (batch_len == 0) {
    io.flush();
    return;
  }

  for (uint32_t i = 0; i < batch_len; ++i) {
    io >> input;

    FeatureExtractor( input, output );

    std::array<float, 11> final_buf;
    f32TensorView<11> final_view{ std::span( final_buf ) };
    ClassifierHead( output, final_view );

    uint8_t argmax = 0;
    float best = final_buf[0];
    for (uint8_t k = 1; k < 11; ++k) {
      if (final_buf[k] > best) {
        best = final_buf[k];
        argmax = k;
      }
    }

    io << argmax;
  }

  io.flush();
}

#endif  // __INFERENCE_H__
