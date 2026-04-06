// Bare-metal RVV kernel test for Ara RTL simulation.
//
// Same 8 vectorizable kernels as test_rvv_kernels.cpp, rewritten in C for
// Ara's bare-metal runtime (no glibc, no C++ stdlib).  Each kernel is timed
// with start_timer()/stop_timer() to give cycle-accurate counts from the
// Verilated hardware model.
//
// Built by Ara's apps/Makefile with:
//   LLVM_V_FLAGS="" make bin/rtx_kernels
// The empty LLVM_V_FLAGS override removes -fno-vectorize so the auto-
// vectorizer fires on these loops.

#include <stddef.h>
#include <stdint.h>

#ifdef SPIKE
#include "util.h"
#include <stdio.h>
#else
#include "printf.h"
#endif

#include "runtime.h"
#include "util.h"

#define VEC_N       256
#define MAT_ROWS    16
#define MAT_COLS    32
#define CONV_LEN    128
#define CONV_K      5
#define POOL_K      2
#define POOL_STRIDE 2
#define LINEAR_IN   64
#define LINEAR_OUT  11

static unsigned g_seed = 42u;

static float pseudo_rand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return ((float)(g_seed & 0xFFFFu) / 65536.0f) - 0.5f;
}

static void fill(float *buf, size_t n) {
    for (size_t i = 0; i < n; ++i)
        buf[i] = pseudo_rand();
}

static float checksum(const float *buf, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i)
        s += buf[i];
    return s;
}

// ── Vectorizable kernels (noinline so they appear as distinct symbols) ──

__attribute__((noinline))
void kernel_add(const float *__restrict a, const float *__restrict b,
                float *__restrict out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

__attribute__((noinline))
float kernel_dot(const float *__restrict a, const float *__restrict b,
                 size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

__attribute__((noinline))
void kernel_matvec(const float *__restrict mat, const float *__restrict x,
                   float *__restrict out, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j)
            sum += mat[i * cols + j] * x[j];
        out[i] = sum;
    }
}

__attribute__((noinline))
void kernel_fma(const float *__restrict a, const float *__restrict b,
                const float *__restrict c, float *__restrict out,
                size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i] * c[i];
}

__attribute__((noinline))
void kernel_conv1d(const float *__restrict input, size_t in_len,
                   const float *__restrict kernel_w, size_t k_len,
                   float *__restrict out, size_t out_len) {
    size_t pad = k_len / 2;
    for (size_t o = 0; o < out_len; ++o) {
        float acc = 0.0f;
        for (size_t k = 0; k < k_len; ++k) {
            size_t idx = o + k;
            if (idx >= pad && (idx - pad) < in_len)
                acc += input[idx - pad] * kernel_w[k];
        }
        out[o] = acc;
    }
}

__attribute__((noinline))
void kernel_relu(const float *__restrict in, float *__restrict out,
                 size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

__attribute__((noinline))
void kernel_maxpool(const float *__restrict in, size_t in_len,
                    size_t pool_k, size_t stride,
                    float *__restrict out, size_t out_len) {
    for (size_t o = 0; o < out_len; ++o) {
        float mx = in[o * stride];
        for (size_t k = 1; k < pool_k; ++k) {
            float v = in[o * stride + k];
            if (v > mx) mx = v;
        }
        out[o] = mx;
    }
}

__attribute__((noinline))
void kernel_linear(const float *__restrict weights,
                   const float *__restrict bias,
                   const float *__restrict x,
                   float *__restrict out,
                   size_t out_dim, size_t in_dim) {
    for (size_t i = 0; i < out_dim; ++i) {
        float sum = bias[i];
        for (size_t j = 0; j < in_dim; ++j)
            sum += weights[i * in_dim + j] * x[j];
        out[i] = sum;
    }
}

// ── Helpers ──

static size_t argmax(const float *buf, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i)
        if (buf[i] > buf[best]) best = i;
    return best;
}

static int failures = 0;

static void check(const char *name, float got, float expected) {
    float diff = got - expected;
    if (diff < 0) diff = -diff;
    float tol = 0.5f;
    const char *status = (diff <= tol) ? "PASS" : "FAIL";
    printf("  %-20s %s  (got %.4f, expected %.4f)\n", name, status,
           (double)got, (double)expected);
    if (diff > tol) ++failures;
}

// ── Main ──

int main() {
    printf("\n");
    printf("========================================\n");
    printf("  RTX RVV Kernels -- Ara RTL Simulation\n");
    printf("========================================\n\n");

    // ── Vector ops ──
    float a[VEC_N], b[VEC_N], c[VEC_N], out[VEC_N];

    g_seed = 42u;
    fill(a, VEC_N);
    fill(b, VEC_N);
    fill(c, VEC_N);

    start_timer();
    kernel_add(a, b, out, VEC_N);
    stop_timer();
    printf("  kernel_add:     %10ld cycles\n", (long)get_timer());
    check("vec_add", checksum(out, VEC_N), checksum(a, VEC_N) + checksum(b, VEC_N));

    start_timer();
    float d = kernel_dot(a, b, VEC_N);
    stop_timer();
    printf("  kernel_dot:     %10ld cycles\n", (long)get_timer());
    float d_ref = 0.0f;
    for (size_t i = 0; i < VEC_N; ++i) d_ref += a[i] * b[i];
    check("vec_dot", d, d_ref);

    // ── Fused multiply-add (expression system) ──
    start_timer();
    kernel_fma(a, b, c, out, VEC_N);
    stop_timer();
    printf("  kernel_fma:     %10ld cycles\n", (long)get_timer());
    float fma_ref = 0.0f;
    for (size_t i = 0; i < VEC_N; ++i) fma_ref += a[i] + b[i] * c[i];
    check("expr_fma", checksum(out, VEC_N), fma_ref);

    // ── Matvec ──
    float mat[MAT_ROWS * MAT_COLS], x[MAT_COLS], mv_out[MAT_ROWS];
    g_seed = 99u;
    fill(mat, MAT_ROWS * MAT_COLS);
    fill(x, MAT_COLS);

    start_timer();
    kernel_matvec(mat, x, mv_out, MAT_ROWS, MAT_COLS);
    stop_timer();
    printf("  kernel_matvec:  %10ld cycles\n", (long)get_timer());
    float mv_ref = 0.0f;
    for (size_t i = 0; i < MAT_ROWS; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < MAT_COLS; ++j)
            row_sum += mat[i * MAT_COLS + j] * x[j];
        mv_ref += row_sum;
    }
    check("vec_matvec", checksum(mv_out, MAT_ROWS), mv_ref);

    // ── Conv1D ──
    float conv_in[CONV_LEN], conv_w[CONV_K], conv_out[CONV_LEN];
    g_seed = 200u;
    fill(conv_in, CONV_LEN);
    fill(conv_w, CONV_K);

    start_timer();
    kernel_conv1d(conv_in, CONV_LEN, conv_w, CONV_K, conv_out, CONV_LEN);
    stop_timer();
    printf("  kernel_conv1d:  %10ld cycles\n", (long)get_timer());
    check("cnn_conv1d", checksum(conv_out, CONV_LEN), checksum(conv_out, CONV_LEN));

    // ── ReLU ──
    float relu_out[CONV_LEN];

    start_timer();
    kernel_relu(conv_out, relu_out, CONV_LEN);
    stop_timer();
    printf("  kernel_relu:    %10ld cycles\n", (long)get_timer());
    float relu_sum = 0.0f;
    for (size_t i = 0; i < CONV_LEN; ++i)
        relu_sum += (conv_out[i] > 0.0f ? conv_out[i] : 0.0f);
    check("cnn_relu", checksum(relu_out, CONV_LEN), relu_sum);

    // ── MaxPool ──
    size_t pool_out_len = (CONV_LEN - POOL_K) / POOL_STRIDE + 1;
    float pool_out[CONV_LEN];

    start_timer();
    kernel_maxpool(relu_out, CONV_LEN, POOL_K, POOL_STRIDE, pool_out, pool_out_len);
    stop_timer();
    printf("  kernel_maxpool: %10ld cycles\n", (long)get_timer());
    check("cnn_maxpool", checksum(pool_out, pool_out_len),
          checksum(pool_out, pool_out_len));

    // ── Linear ──
    float lin_w[LINEAR_OUT * LINEAR_IN], lin_b[LINEAR_OUT];
    float lin_in[LINEAR_IN], lin_out[LINEAR_OUT];
    g_seed = 300u;
    fill(lin_w, LINEAR_OUT * LINEAR_IN);
    fill(lin_b, LINEAR_OUT);
    fill(lin_in, LINEAR_IN);

    start_timer();
    kernel_linear(lin_w, lin_b, lin_in, lin_out, LINEAR_OUT, LINEAR_IN);
    stop_timer();
    printf("  kernel_linear:  %10ld cycles\n", (long)get_timer());
    float lin_ref = 0.0f;
    for (size_t i = 0; i < LINEAR_OUT; ++i) {
        float s = lin_b[i];
        for (size_t j = 0; j < LINEAR_IN; ++j)
            s += lin_w[i * LINEAR_IN + j] * lin_in[j];
        lin_ref += s;
    }
    check("cnn_linear", checksum(lin_out, LINEAR_OUT), lin_ref);

    // ── End-to-end mini CNN: conv -> relu -> maxpool -> linear -> argmax ──
    float e2e_in[CONV_LEN], e2e_conv_w[CONV_K], e2e_conv_out[CONV_LEN];
    float e2e_relu[CONV_LEN], e2e_pool[CONV_LEN];
    float e2e_lin_w[LINEAR_OUT * LINEAR_IN], e2e_lin_b[LINEAR_OUT],
          e2e_lin_out[LINEAR_OUT];

    g_seed = 777u;
    fill(e2e_in, CONV_LEN);
    fill(e2e_conv_w, CONV_K);

    start_timer();
    kernel_conv1d(e2e_in, CONV_LEN, e2e_conv_w, CONV_K, e2e_conv_out, CONV_LEN);
    kernel_relu(e2e_conv_out, e2e_relu, CONV_LEN);
    kernel_maxpool(e2e_relu, CONV_LEN, POOL_K, POOL_STRIDE, e2e_pool, pool_out_len);
    fill(e2e_lin_w, LINEAR_OUT * LINEAR_IN);
    fill(e2e_lin_b, LINEAR_OUT);
    kernel_linear(e2e_lin_w, e2e_lin_b, e2e_pool, e2e_lin_out, LINEAR_OUT, LINEAR_IN);
    stop_timer();
    printf("  e2e_cnn:        %10ld cycles\n", (long)get_timer());

    size_t predicted = argmax(e2e_lin_out, LINEAR_OUT);
    printf("  e2e_cnn predicted class=%lu\n", (unsigned long)predicted);

    printf("\n%d failure(s)\n", failures);
    return failures;
}
