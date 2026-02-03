#include <iostream>
#include <algorithm>
#include "ExprFunctions.h"

#include <span>

template <size_t Dimension>
struct Vector {
  std::span<float, Dimension> _data;

  constexpr Vector( std::array<float, Dimension>& input ) : _data(input) {}

  static constexpr std::extents<size_t, Dimension> extents{};
  static constexpr size_t iter_size() { return Dimension; }

  using value_type = float;

  constexpr const value_type& access(size_t i) const {
    return _data[i];
  }

  constexpr value_type& access(size_t i) {
    return _data[i];
  }

};

inline std::array<float, 16> left;
inline std::array<float, 16> right;
inline std::array<float, 16> buf;

int main() {
  std::cout << "Hello from RISC-V (or native) via C++23.\n";

  std::ranges::iota(left, 0);
  std::ranges::fill(right, 14);

  constexpr Vector<16> lv{left};
  constexpr Vector<16> rv{right};
  
  auto tmp = lv + rv;

  constexpr Vector<16> res{buf};

  in_place_eval( tmp, res );

  return 0;
}

/*
Resulting disassembly:
We are just loading and printing values. The compiler is able to completely see
we are doing left + right = res, so it just does the work here completely at compile-time.
0000000000001050 <main>:
    1050:       48 83 ec 08             sub    $0x8,%rsp
    1054:       ba 29 00 00 00          mov    $0x29,%edx
    1059:       48 8d 35 a8 0f 00 00    lea    0xfa8(%rip),%rsi        # 2008 <_IO_stdin_used+0x8>
    1060:       48 8d 3d d9 2f 00 00    lea    0x2fd9(%rip),%rdi        # 4040 <_ZSt4cout@GLIBCXX_3.4>
    1067:       e8 c4 ff ff ff          call   1030 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    106c:       0f 28 05 cd 0f 00 00    movaps 0xfcd(%rip),%xmm0        # 2040 <_IO_stdin_used+0x40>
    1073:       31 c0                   xor    %eax,%eax
    1075:       0f 11 05 64 31 00 00    movups %xmm0,0x3164(%rip)        # 41e0 <left>
    107c:       0f 28 05 cd 0f 00 00    movaps 0xfcd(%rip),%xmm0        # 2050 <_IO_stdin_used+0x50>
    1083:       0f 11 05 66 31 00 00    movups %xmm0,0x3166(%rip)        # 41f0 <left+0x10>
    108a:       0f 28 05 cf 0f 00 00    movaps 0xfcf(%rip),%xmm0        # 2060 <_IO_stdin_used+0x60>
    1091:       0f 11 05 68 31 00 00    movups %xmm0,0x3168(%rip)        # 4200 <left+0x20>
    1098:       0f 28 05 d1 0f 00 00    movaps 0xfd1(%rip),%xmm0        # 2070 <_IO_stdin_used+0x70>
    109f:       0f 11 05 6a 31 00 00    movups %xmm0,0x316a(%rip)        # 4210 <left+0x30>
    10a6:       f3 0f 10 05 ca 0f 00    movss  0xfca(%rip),%xmm0        # 2078 <_IO_stdin_used+0x78>
    10ad:       00
    10ae:       0f c6 c0 00             shufps $0x0,%xmm0,%xmm0
    10b2:       0f 11 05 e7 30 00 00    movups %xmm0,0x30e7(%rip)        # 41a0 <right>
    10b9:       0f 11 05 f0 30 00 00    movups %xmm0,0x30f0(%rip)        # 41b0 <right+0x10>
    10c0:       0f 11 05 f9 30 00 00    movups %xmm0,0x30f9(%rip)        # 41c0 <right+0x20>
    10c7:       0f 11 05 02 31 00 00    movups %xmm0,0x3102(%rip)        # 41d0 <right+0x30>
    10ce:       0f 28 05 ab 0f 00 00    movaps 0xfab(%rip),%xmm0        # 2080 <_IO_stdin_used+0x80>
    10d5:       0f 11 05 84 30 00 00    movups %xmm0,0x3084(%rip)        # 4160 <buf>
    10dc:       0f 28 05 ad 0f 00 00    movaps 0xfad(%rip),%xmm0        # 2090 <_IO_stdin_used+0x90>
    10e3:       0f 11 05 86 30 00 00    movups %xmm0,0x3086(%rip)        # 4170 <buf+0x10>
    10ea:       0f 28 05 af 0f 00 00    movaps 0xfaf(%rip),%xmm0        # 20a0 <_IO_stdin_used+0xa0>
    10f1:       0f 11 05 88 30 00 00    movups %xmm0,0x3088(%rip)        # 4180 <buf+0x20>
    10f8:       0f 28 05 b1 0f 00 00    movaps 0xfb1(%rip),%xmm0        # 20b0 <_IO_stdin_used+0xb0>
    10ff:       0f 11 05 8a 30 00 00    movups %xmm0,0x308a(%rip)        # 4190 <buf+0x30>
    1106:       48 83 c4 08             add    $0x8,%rsp
    110a:       c3                      ret
    110b:       0f 1f 44 00 00          nopl   0x0(%rax,%rax,1)

*/