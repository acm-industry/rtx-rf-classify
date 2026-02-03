#include <iostream>
#include <algorithm>
#include "ExprFunctions.h"
#include <random>
#include <print>

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

inline std::random_device rd;
inline std::mt19937 gen(rd());
inline std::uniform_real_distribution<float> dis(-10.0, 10.0);

constexpr auto rand_num = [] { return dis(gen); };

void print_stuff( const std::array<float, 16>& in ) {
  for (size_t i = 0; i < in.size(); ++i) std::print("{}\n", in[i]); 
}

int main() {

  // We have to generate random values so the compiler can't know the values beforehand
  std::ranges::generate( left, rand_num );
  std::ranges::generate( right, rand_num );

  constexpr Vector<16> lv{left};
  constexpr Vector<16> rv{right};
  
  auto tmp = lv + rv;

  Vector<16> res{buf};

  in_place_eval( tmp, res );

  print_stuff(buf);

  return 0;
}

/*
When compiled with -mavx and -O3, the below is the resulting disassm.
it uses 8-byte wide YMM registers, without any explicit instructions to do so.
00000000000023c0 <main>:
    23c0:       55                      push   %rbp
    23c1:       48 8d 3d b8 c5 01 00    lea    0x1c5b8(%rip),%rdi        # 1e980 <left>
    23c8:       48 89 e5                mov    %rsp,%rbp
    23cb:       41 57                   push   %r15
    23cd:       41 56                   push   %r14
    23cf:       4c 8d 35 2a c5 01 00    lea    0x1c52a(%rip),%r14        # 1e900 <buf>
    23d6:       41 55                   push   %r13
    23d8:       4d 8d 7e 40             lea    0x40(%r14),%r15
    23dc:       41 54                   push   %r12
    23de:       53                      push   %rbx
    23df:       48 83 e4 e0             and    $0xffffffffffffffe0,%rsp
    23e3:       48 83 ec 20             sub    $0x20,%rsp
    23e7:       e8 14 08 00 00          call   2c00 <_ZNKSt6ranges13__generate_fnclIRSt5arrayIfLm16EENL8rand_numMUlvE_EEENSt13__conditionalIX14borrowed_rangeIT_EEE4typeIDTcl7__begincl7declvalIRS8_EEEENS_8danglingEEEOS8_T0_.constprop.0.isra.0>
    23ec:       48 8d 3d 4d c5 01 00    lea    0x1c54d(%rip),%rdi        # 1e940 <right>
    23f3:       48 8d 5c 24 10          lea    0x10(%rsp),%rbx
    23f8:       e8 03 08 00 00          call   2c00 <_ZNKSt6ranges13__generate_fnclIRSt5arrayIfLm16EENL8rand_numMUlvE_EEENSt13__conditionalIX14borrowed_rangeIT_EEE4typeIDTcl7__begincl7declvalIRS8_EEEENS_8danglingEEEOS8_T0_.constprop.0.isra.0>
    23fd:       c5 fc 10 05 7b c5 01    vmovups 0x1c57b(%rip),%ymm0        # 1e980 <left>
    2404:       00
    2405:       c5 fc 58 05 33 c5 01    vaddps 0x1c533(%rip),%ymm0,%ymm0        # 1e940 <right>
    240c:       00
    240d:       c5 fc 11 05 eb c4 01    vmovups %ymm0,0x1c4eb(%rip)        # 1e900 <buf>
    2414:       00
    2415:       c5 fc 10 05 83 c5 01    vmovups 0x1c583(%rip),%ymm0        # 1e9a0 <left+0x20>
    241c:       00
    241d:       c5 fc 58 05 3b c5 01    vaddps 0x1c53b(%rip),%ymm0,%ymm0        # 1e960 <right+0x20>
    2424:       00
    2425:       c5 fc 11 05 f3 c4 01    vmovups %ymm0,0x1c4f3(%rip)        # 1e920 <buf+0x20>
    242c:       00
    242d:       c5 f8 77                vzeroupper
    2430:       c4 c1 7a 10 06          vmovss (%r14),%xmm0
    2435:       41 bc 71 00 00 00       mov    $0x71,%r12d
    243b:       49 89 d8                mov    %rbx,%r8
    243e:       49 83 c6 04             add    $0x4,%r14
    2442:       48 8b 3d 37 9d 01 00    mov    0x19d37(%rip),%rdi        # 1c180 <stdout@GLIBC_2.2.5>
    2449:       4c 89 e1                mov    %r12,%rcx
    244c:       be 03 00 00 00          mov    $0x3,%esi
    2451:       48 8d 15 22 3c 01 00    lea    0x13c22(%rip),%rdx        # 1607a <_IO_stdin_used+0x7a>
    2458:       c5 fa 11 44 24 10       vmovss %xmm0,0x10(%rsp)
    245e:       e8 dd 12 01 00          call   13740 <_ZSt17vprint_nonunicodeP8_IO_FILESt17basic_string_viewIcSt11char_traitsIcEESt17basic_format_argsISt20basic_format_contextINSt8__format10_Sink_iterIcEEcEE>
    2463:       4d 39 fe                cmp    %r15,%r14
    2466:       75 c8                   jne    2430 <main+0x70>
    2468:       48 8d 65 d8             lea    -0x28(%rbp),%rsp
    246c:       31 c0                   xor    %eax,%eax
    246e:       5b                      pop    %rbx
    246f:       41 5c                   pop    %r12
    2471:       41 5d                   pop    %r13
    2473:       41 5e                   pop    %r14
    2475:       41 5f                   pop    %r15
    2477:       5d                      pop    %rbp
    2478:       c3                      ret
    2479:       0f 1f 80 00 00 00 00    nopl   0x0(%rax)
*/