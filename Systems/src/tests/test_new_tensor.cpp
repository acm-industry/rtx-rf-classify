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

  std::ranges::generate( x.flat_view(), [] { return dist(gen); } );

  DynTensor<float, std::extents<size_t, 16>, decltype(alloc)> y = x.clone();

  y[ idist(gen) ] = -100;

  std::cout << "x: ";
  for (float value : x.flat_view()) std::cout << value << ' ';
  std::cout << '\n';

  std::cout << "y: ";
  for (float value : y.flat_view()) std::cout << value << ' ';
  std::cout << '\n';
  
  std::cout << std::boolalpha;
  for (bool is_same : abs(x.as_view() - y.as_view()) | std::views::transform([](float val){ return val < 1e-3; }) ) 
    std::cout << is_same << ' ';
  
  std::cout << '\n';
 
  return 0;
}