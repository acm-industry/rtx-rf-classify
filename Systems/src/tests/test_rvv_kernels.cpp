#include <cstddef>
#include <cstdio>
#include <cmath>
#include <cstring>

// Self-contained vectorizable kernels for RVV verification.
// Each kernel uses simple loops that GCC can auto-vectorize with
//   -march=rv64gcv -O3 -fno-vect-cost-model -ffast-math
// They mirror the operations in vector_ops (add/dot/matmul/matvec),
// the expression system (fused multiply-add), and CNN inference layers
// (conv1d, relu, maxpool, linear).
//
// __attribute__((noinline)) prevents GCC from inlining these into main(),
// keeping each kernel as a distinct labeled function in the assembly output
// so the per-kernel RVV instruction grep can find them.

static constexpr size_t VEC_N = 256;
static constexpr size_t MAT_ROWS = 16;
static constexpr size_t MAT_COLS = 32;
static constexpr size_t CONV_LEN = 128;
static constexpr size_t CONV_K = 5;
static constexpr size_t POOL_K = 2;
static constexpr size_t POOL_STRIDE = 2;
static constexpr size_t LINEAR_IN = 64;
static constexpr size_t LINEAR_OUT = 11;

static unsigned g_seed = 42u;
static float pseudo_rand() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (static_cast<float>(g_seed & 0xFFFFu) / 65536.0f) - 0.5f;
}

static void fill(float* buf, size_t n) {
    for (size_t i = 0; i < n; ++i) buf[i] = pseudo_rand();
}

static float checksum(const float* buf, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += buf[i];
    return s;
}

// ── vec::add equivalent ──
__attribute__((noinline))
void kernel_add(const float* __restrict a, const float* __restrict b,
                float* __restrict out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

// ── vec::dot equivalent ──
__attribute__((noinline))
float kernel_dot(const float* __restrict a, const float* __restrict b,
                 size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

// ── vec::matvec equivalent ──
__attribute__((noinline))
void kernel_matvec(const float* __restrict mat, const float* __restrict x,
                   float* __restrict out, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j)
            sum += mat[i * cols + j] * x[j];
        out[i] = sum;
    }
}

// ── Expression system fused multiply-add: out = a + b * c ──
__attribute__((noinline))
void kernel_fma(const float* __restrict a, const float* __restrict b,
                const float* __restrict c, float* __restrict out,
                size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i] * c[i];
}

// ── CNN layer: 1D convolution (same padding via zero-pad) ──
__attribute__((noinline))
void kernel_conv1d(const float* __restrict input, size_t in_len,
                   const float* __restrict kernel_w, size_t k_len,
                   float* __restrict out, size_t out_len) {
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

// ── CNN layer: ReLU activation ──
__attribute__((noinline))
void kernel_relu(const float* __restrict in, float* __restrict out,
                 size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

// ── CNN layer: 1D max-pooling ──
__attribute__((noinline))
void kernel_maxpool(const float* __restrict in, size_t in_len,
                    size_t pool_k, size_t stride,
                    float* __restrict out, size_t out_len) {
    for (size_t o = 0; o < out_len; ++o) {
        float mx = in[o * stride];
        for (size_t k = 1; k < pool_k; ++k) {
            float v = in[o * stride + k];
            if (v > mx) mx = v;
        }
        out[o] = mx;
    }
}

// ── CNN layer: linear (matrix-vector + bias) ──
__attribute__((noinline))
void kernel_linear(const float* __restrict weights,
                   const float* __restrict bias,
                   const float* __restrict x,
                   float* __restrict out,
                   size_t out_dim, size_t in_dim) {
    for (size_t i = 0; i < out_dim; ++i) {
        float sum = bias[i];
        for (size_t j = 0; j < in_dim; ++j)
            sum += weights[i * in_dim + j] * x[j];
        out[i] = sum;
    }
}

// ── Argmax helper ──
static size_t argmax(const float* buf, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i)
        if (buf[i] > buf[best]) best = i;
    return best;
}

// ── Test harness ──

static int failures = 0;

static void check(const char* name, float got, float expected, float tol = 0.5f) {
    float diff = got - expected;
    if (diff < 0) diff = -diff;
    const char* status = (diff <= tol) ? "PASS" : "FAIL";
    std::printf("%-20s %s  (got %.6f, expected %.6f)\n", name, status, got, expected);
    if (diff > tol) ++failures;
}

int main() {
    // ──── Vector ops ────
    float a[VEC_N], b[VEC_N], c[VEC_N], out[VEC_N];

    g_seed = 42u;
    fill(a, VEC_N);
    fill(b, VEC_N);
    fill(c, VEC_N);

    kernel_add(a, b, out, VEC_N);
    check("vec_add", checksum(out, VEC_N), checksum(a, VEC_N) + checksum(b, VEC_N));

    float d = kernel_dot(a, b, VEC_N);
    float d_ref = 0.0f;
    for (size_t i = 0; i < VEC_N; ++i) d_ref += a[i] * b[i];
    check("vec_dot", d, d_ref);

    // ──── Fused multiply-add (expression system) ────
    kernel_fma(a, b, c, out, VEC_N);
    float fma_ref = 0.0f;
    for (size_t i = 0; i < VEC_N; ++i) fma_ref += a[i] + b[i] * c[i];
    check("expr_fma", checksum(out, VEC_N), fma_ref);

    // ──── Matvec ────
    float mat[MAT_ROWS * MAT_COLS], x[MAT_COLS], mv_out[MAT_ROWS];
    g_seed = 99u;
    fill(mat, MAT_ROWS * MAT_COLS);
    fill(x, MAT_COLS);
    kernel_matvec(mat, x, mv_out, MAT_ROWS, MAT_COLS);
    float mv_ref = 0.0f;
    for (size_t i = 0; i < MAT_ROWS; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < MAT_COLS; ++j)
            row_sum += mat[i * MAT_COLS + j] * x[j];
        mv_ref += row_sum;
    }
    check("vec_matvec", checksum(mv_out, MAT_ROWS), mv_ref);

    // ──── Conv1D ────
    float conv_in[CONV_LEN], conv_w[CONV_K], conv_out[CONV_LEN];
    g_seed = 200u;
    fill(conv_in, CONV_LEN);
    fill(conv_w, CONV_K);
    kernel_conv1d(conv_in, CONV_LEN, conv_w, CONV_K, conv_out, CONV_LEN);
    check("cnn_conv1d", checksum(conv_out, CONV_LEN), checksum(conv_out, CONV_LEN));

    // ──── ReLU ────
    float relu_out[CONV_LEN];
    kernel_relu(conv_out, relu_out, CONV_LEN);
    float relu_sum = 0.0f;
    for (size_t i = 0; i < CONV_LEN; ++i)
        relu_sum += (conv_out[i] > 0.0f ? conv_out[i] : 0.0f);
    check("cnn_relu", checksum(relu_out, CONV_LEN), relu_sum);

    // ──── MaxPool ────
    size_t pool_out_len = (CONV_LEN - POOL_K) / POOL_STRIDE + 1;
    float pool_out[CONV_LEN];
    kernel_maxpool(relu_out, CONV_LEN, POOL_K, POOL_STRIDE, pool_out, pool_out_len);
    check("cnn_maxpool", checksum(pool_out, pool_out_len),
          checksum(pool_out, pool_out_len));

    // ──── Linear ────
    float lin_w[LINEAR_OUT * LINEAR_IN], lin_b[LINEAR_OUT];
    float lin_in[LINEAR_IN], lin_out[LINEAR_OUT];
    g_seed = 300u;
    fill(lin_w, LINEAR_OUT * LINEAR_IN);
    fill(lin_b, LINEAR_OUT);
    fill(lin_in, LINEAR_IN);
    kernel_linear(lin_w, lin_b, lin_in, lin_out, LINEAR_OUT, LINEAR_IN);
    float lin_ref = 0.0f;
    for (size_t i = 0; i < LINEAR_OUT; ++i) {
        float s = lin_b[i];
        for (size_t j = 0; j < LINEAR_IN; ++j) s += lin_w[i * LINEAR_IN + j] * lin_in[j];
        lin_ref += s;
    }
    check("cnn_linear", checksum(lin_out, LINEAR_OUT), lin_ref);

    // ──── End-to-end mini CNN: conv -> relu -> maxpool -> linear -> argmax ────
    float e2e_in[CONV_LEN], e2e_conv_w[CONV_K], e2e_conv_out[CONV_LEN];
    float e2e_relu[CONV_LEN], e2e_pool[CONV_LEN];
    float e2e_lin_w[LINEAR_OUT * LINEAR_IN], e2e_lin_b[LINEAR_OUT], e2e_lin_out[LINEAR_OUT];

    g_seed = 777u;
    fill(e2e_in, CONV_LEN);
    fill(e2e_conv_w, CONV_K);
    kernel_conv1d(e2e_in, CONV_LEN, e2e_conv_w, CONV_K, e2e_conv_out, CONV_LEN);
    kernel_relu(e2e_conv_out, e2e_relu, CONV_LEN);
    kernel_maxpool(e2e_relu, CONV_LEN, POOL_K, POOL_STRIDE, e2e_pool, pool_out_len);
    fill(e2e_lin_w, LINEAR_OUT * LINEAR_IN);
    fill(e2e_lin_b, LINEAR_OUT);
    kernel_linear(e2e_lin_w, e2e_lin_b, e2e_pool, e2e_lin_out, LINEAR_OUT, LINEAR_IN);
    size_t predicted = argmax(e2e_lin_out, LINEAR_OUT);
    std::printf("%-20s class=%zu\n", "e2e_cnn", predicted);

    std::printf("\n%d failure(s)\n", failures);
    return failures;
}
