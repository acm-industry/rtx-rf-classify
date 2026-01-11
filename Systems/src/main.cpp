#include "Expression.hpp"
#include <iostream>
using namespace std;

int main() {
    using namespace rtxsys;

    constexpr ScalarExpr<float> x = 3.2f;
    constexpr ScalarExpr<float> y = 4.9f;

    constexpr auto expr = sin(x) + y * x;

    using long_tpye = decltype(expr);

    ScalarExpr<float> result{};

    in_place_eval( expr, result );

    std::cout << result.value << '\n';

    return 0;    
}

/*
RESULTING DISASSEMBLY ( objdump -d )

0000000000001090 <main>:
    1090:       48 83 ec 18             sub    $0x18,%rsp
    1094:       c5 fb 10 05 6c 0f 00    vmovsd 0xf6c(%rip),%xmm0        # 2008 <_IO_stdin_used+0x8>
    109b:       00
    109c:       48 8d 3d 9d 2f 00 00    lea    0x2f9d(%rip),%rdi        # 4040 <_ZSt4cout@GLIBCXX_3.4>
    10a3:       e8 c8 ff ff ff          call   1070 <_ZNSo9_M_insertIdEERSoT_@plt>
    10a8:       c6 44 24 0f 0a          movb   $0xa,0xf(%rsp)
    10ad:       48 89 c7                mov    %rax,%rdi
    10b0:       48 8b 00                mov    (%rax),%rax
    10b3:       48 8b 40 e8             mov    -0x18(%rax),%rax
    10b7:       48 83 7c 07 10 00       cmpq   $0x0,0x10(%rdi,%rax,1)
    10bd:       74 16                   je     10d5 <main+0x45>
    10bf:       48 8d 74 24 0f          lea    0xf(%rsp),%rsi
    10c4:       ba 01 00 00 00          mov    $0x1,%edx
    10c9:       e8 82 ff ff ff          call   1050 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    10ce:       31 c0                   xor    %eax,%eax
    10d0:       48 83 c4 18             add    $0x18,%rsp
    10d4:       c3                      ret
    10d5:       be 0a 00 00 00          mov    $0xa,%esi
    10da:       e8 51 ff ff ff          call   1030 <_ZNSo3putEc@plt>
    10df:       eb ed                   jmp    10ce <main+0x3e>
    10e1:       66 66 2e 0f 1f 84 00    data16 cs nopw 0x0(%rax,%rax,1)
    10e8:       00 00 00 00
    10ec:       0f 1f 40 00             nopl   0x0(%rax)

*/