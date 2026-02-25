#include <iostream>
#include "tensor.h"
#include "memorybuffer.h"
#include <random>
#include <ranges>
#include "ExprSystem/ExprFunctions.h"

std::random_device rd;
std::mt19937 gen{rd()};
std::uniform_real_distribution<float> dist{1.0, 5.0};
std::uniform_int_distribution<size_t> idist{0, 15};


int main() {
  alignas(32) std::array<std::byte, 512 * sizeof(float)> memory;
  MemoryBuffer buf( memory );
  auto alloc = buf.get_allocator<float, 32>();

  DynTensor<float, std::extents<size_t, 4, 4>, decltype(alloc)> x(alloc);
  DynTensor<float, std::extents<size_t, 4, 4>, decltype(alloc)> y(alloc);
  DynTensor<float, std::extents<size_t, 4, 4>, decltype(alloc)> z(alloc);

  std::ranges::generate( x.flat_view(), [] { return dist(gen); } );
  std::ranges::generate( y.flat_view(), [] { return dist(gen); } );

  auto& x_view = x.as_view();
  auto& y_view = y.as_view();

  z = ( exp(x_view + y_view) - exp(-x_view - y_view) );

  std::cout << x.prettyPrint() << '\n';
  std::cout << y.prettyPrint() << '\n';
  std::cout << z.prettyPrint() << '\n';
 
  return 0;
}