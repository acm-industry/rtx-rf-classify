#include <iostream>
#include "tensor.h"
#include "convolve.h"
#include "batchnorm.h"
#include <random>
#include <ranges>
#include <algorithm>
#include "ExprSystem/Broadcast.h"
#include "ExprSystem/Expression.h"
#include "ExprSystem/Scalar.h"
#include "ExprSystem/ExprFunctions.h"
#include "memorybuffer.h"
#include "maxpool.h"
#include "avgpool.h"
#include "operators/unary_ops.h"
#include "maweights.cpp"
#include "network.h"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(-100, 100);

template <size_t... idxs> requires ( sizeof...(idxs) > 0 )
using f32Tensor = DynTensor<float, std::extents<size_t, idxs...>, MemoryBuffer::Allocator<float, 32>>;

template <size_t... idxs> requires ( sizeof...(idxs) > 0 )
using f32TensorView = TensorBase<float, std::extents<size_t, idxs...>>;


template <size_t padding, FixedExtent E1, FixedExtent E2, FixedExtent E3, FixedExtent E4> requires ( E1::rank() == 2 && E2::rank() == 3 && E3::rank() == 1 && E4::rank() == 2 )
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

void FeatureExtractor( const f32TensorView<3, 128>& input, f32TensorView<128, 32>& out ) {
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

  auto slice = conv_weights_layer1[0,0, 0];
  using demo = decltype(slice);

  f32TensorView<64> conv_bias_weights_layer1{ std::span<float, 64>(weights_conv_1d_bias_l1, 64) };

  BroadcastConvolve<2>( input, conv_weights_layer1, conv_bias_weights_layer1, conv_out_layer1 );
   
  f32TensorView<4, 64> bn_weights_layer1{ std::span<float, 4 * 64>( weights_bn_1d_l1, 4 * 64 ) };

  BatchNorm1DInPlace( conv_out_layer1, bn_weights_layer1, conv_out_layer1 );

  ops::ReluOp relu_op;
  relu_op(conv_out_layer1, conv_out_layer1);

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

  relu_op(conv_out_layer2, conv_out_layer2);

  buf2.reset();

  f32Tensor<128, 32> mp_out_layer2(allocator2);

  for (size_t i = 0; i < 128; ++i) {
    auto outvec = mp_out_layer2[i];
    
    using demo = decltype(outvec);

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

  relu_op(conv_out_layer3, conv_out_layer3);
 
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

  relu_op(conv_out_layer4, out);

}

void ClassifierHead(const f32TensorView<128, 32>& input, f32TensorView<11>& out ) {
  
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

int main() {

  static constexpr size_t INPUT_OUTPUT_ALLOC_BYTES = ( 3 * 128 + 128 * 32 ) * sizeof(float) + 256;
  MemoryBuffer buf( INPUT_OUTPUT_ALLOC_BYTES ); // Alloc enough for input/output tensor plus some wiggle room
  auto alloc = buf.get_allocator<float, 32>();

  f32Tensor<3, 128> input(alloc);
  f32Tensor<128, 32> output(alloc);

  Server server(8080);

  std::cout << "Listening on port 8080.\n";

  for (;;) {
    Connection conn = server.accept();

    std::cout << "Recieved connection. Running compute...\n";

    // MVP TCP protocol: 4 bytes on batch len, then batch_len instances of 3 * 128 bytes.

    unsigned int batch_len_net;
    conn.recv( std::span<std::byte>( reinterpret_cast<std::byte*>(&batch_len_net), sizeof(unsigned int) ) );
    unsigned int batch_len = ntohl( batch_len_net );

    if ( batch_len == 0 ) continue;

    try {

      for (unsigned int i = 0; i < batch_len; ++i) {
        conn.recv( std::span<std::byte>( reinterpret_cast<std::byte*>(input.data()), 3 * 128 * sizeof(float)) );

        FeatureExtractor( input, output );

        std::array<float, 11> final_buf;

        f32TensorView<11> final{ std::span( final_buf ) };

        ClassifierHead( output, final );

        unsigned char argmax = std::ranges::distance(final.begin(), std::ranges::max_element( final ));

        conn.send( std::span<std::byte>( reinterpret_cast<std::byte*>(&argmax) , 1 ) );
      }


      std::cout << "Batch sent.\n";

    } catch ( const std::runtime_error& err ) {
      std::cout << "Connection closed. Killing compute.\n";
    }

  }

  return 0;
}
