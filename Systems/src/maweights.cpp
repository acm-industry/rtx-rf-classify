#include <cstddef>

#if WEIGHTS_FP16
#include "fp16_decode.h"
#endif

// Store all of the weights as binaries and link them

extern const unsigned char weights_conv_1d_l1_start[];
extern const unsigned char weights_conv_1d_bias_l1_start[];
extern const unsigned char weights_bn_1d_l1_start[];

extern const unsigned char weights_conv_1d_l2_start[];
extern const unsigned char weights_conv_1d_bias_l2_start[];
extern const unsigned char weights_bn_1d_l2_start[];

extern const unsigned char weights_conv_1d_l3_start[];
extern const unsigned char weights_conv_1d_bias_l3_start[];
extern const unsigned char weights_bn_1d_l3_start[];

extern const unsigned char weights_conv_1d_l4_start[];
extern const unsigned char weights_conv_1d_bias_l4_start[];
extern const unsigned char weights_bn_1d_l4_start[];

extern const unsigned char linear_mat_1_start[];
extern const unsigned char linear_add_1_start[];

extern const unsigned char linear_mat_2_start[];
extern const unsigned char linear_add_2_start[];

#if WEIGHTS_FP16

namespace {

constexpr size_t E_CONV1 = 64 * 3 * 5;
constexpr size_t E_BIAS1 = 64;
constexpr size_t E_BN1 = 4 * 64;

constexpr size_t E_CONV2 = 128 * 64 * 5;
constexpr size_t E_BIAS2 = 128;
constexpr size_t E_BN2 = 4 * 128;

constexpr size_t E_CONV3 = 128 * 128 * 3;
constexpr size_t E_BIAS3 = 128;
constexpr size_t E_BN3 = 4 * 128;

constexpr size_t E_CONV4 = 128 * 128 * 3;
constexpr size_t E_BIAS4 = 128;
constexpr size_t E_BN4 = 4 * 128;

constexpr size_t E_LIN1 = 128 * 128;
constexpr size_t E_LIN1B = 128;
constexpr size_t E_LIN2 = 11 * 128;
constexpr size_t E_LIN2B = 11;

void decode_fp16_blob(const unsigned char* start, float* dst, size_t n) {
    const auto* p = reinterpret_cast<const uint16_t*>(start);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = rtx::fp16::half_to_float(p[i]);
    }
}

alignas(32) static float buf_conv_1d_l1[E_CONV1];
alignas(32) static float buf_conv_1d_bias_l1[E_BIAS1];
alignas(32) static float buf_bn_1d_l1[E_BN1];

alignas(32) static float buf_conv_1d_l2[E_CONV2];
alignas(32) static float buf_conv_1d_bias_l2[E_BIAS2];
alignas(32) static float buf_bn_1d_l2[E_BN2];

alignas(32) static float buf_conv_1d_l3[E_CONV3];
alignas(32) static float buf_conv_1d_bias_l3[E_BIAS3];
alignas(32) static float buf_bn_1d_l3[E_BN3];

alignas(32) static float buf_conv_1d_l4[E_CONV4];
alignas(32) static float buf_conv_1d_bias_l4[E_BIAS4];
alignas(32) static float buf_bn_1d_l4[E_BN4];

alignas(32) static float buf_linear_mat_1[E_LIN1];
alignas(32) static float buf_linear_add_1[E_LIN1B];

alignas(32) static float buf_linear_mat_2[E_LIN2];
alignas(32) static float buf_linear_add_2[E_LIN2B];

} // namespace

float* weights_conv_1d_l1 = buf_conv_1d_l1;
float* weights_conv_1d_bias_l1 = buf_conv_1d_bias_l1;
float* weights_bn_1d_l1 = buf_bn_1d_l1;

float* weights_conv_1d_l2 = buf_conv_1d_l2;
float* weights_conv_1d_bias_l2 = buf_conv_1d_bias_l2;
float* weights_bn_1d_l2 = buf_bn_1d_l2;

float* weights_conv_1d_l3 = buf_conv_1d_l3;
float* weights_conv_1d_bias_l3 = buf_conv_1d_bias_l3;
float* weights_bn_1d_l3 = buf_bn_1d_l3;

float* weights_conv_1d_l4 = buf_conv_1d_l4;
float* weights_conv_1d_bias_l4 = buf_conv_1d_bias_l4;
float* weights_bn_1d_l4 = buf_bn_1d_l4;

float* linear_mat_1 = buf_linear_mat_1;
float* linear_add_1 = buf_linear_add_1;

float* linear_mat_2 = buf_linear_mat_2;
float* linear_add_2 = buf_linear_add_2;

void init_fp16_weights() {
    decode_fp16_blob(weights_conv_1d_l1_start, buf_conv_1d_l1, E_CONV1);
    decode_fp16_blob(weights_conv_1d_bias_l1_start, buf_conv_1d_bias_l1, E_BIAS1);
    decode_fp16_blob(weights_bn_1d_l1_start, buf_bn_1d_l1, E_BN1);

    decode_fp16_blob(weights_conv_1d_l2_start, buf_conv_1d_l2, E_CONV2);
    decode_fp16_blob(weights_conv_1d_bias_l2_start, buf_conv_1d_bias_l2, E_BIAS2);
    decode_fp16_blob(weights_bn_1d_l2_start, buf_bn_1d_l2, E_BN2);

    decode_fp16_blob(weights_conv_1d_l3_start, buf_conv_1d_l3, E_CONV3);
    decode_fp16_blob(weights_conv_1d_bias_l3_start, buf_conv_1d_bias_l3, E_BIAS3);
    decode_fp16_blob(weights_bn_1d_l3_start, buf_bn_1d_l3, E_BN3);

    decode_fp16_blob(weights_conv_1d_l4_start, buf_conv_1d_l4, E_CONV4);
    decode_fp16_blob(weights_conv_1d_bias_l4_start, buf_conv_1d_bias_l4, E_BIAS4);
    decode_fp16_blob(weights_bn_1d_l4_start, buf_bn_1d_l4, E_BN4);

    decode_fp16_blob(linear_mat_1_start, buf_linear_mat_1, E_LIN1);
    decode_fp16_blob(linear_add_1_start, buf_linear_add_1, E_LIN1B);

    decode_fp16_blob(linear_mat_2_start, buf_linear_mat_2, E_LIN2);
    decode_fp16_blob(linear_add_2_start, buf_linear_add_2, E_LIN2B);
}

#else

float* weights_conv_1d_l1 = (float*)weights_conv_1d_l1_start;
float* weights_conv_1d_bias_l1 = (float*)weights_conv_1d_bias_l1_start;
float* weights_bn_1d_l1 = (float*)weights_bn_1d_l1_start;

float* weights_conv_1d_l2 = (float*)weights_conv_1d_l2_start;
float* weights_conv_1d_bias_l2 = (float*)weights_conv_1d_bias_l2_start;
float* weights_bn_1d_l2 = (float*)weights_bn_1d_l2_start;

float* weights_conv_1d_l3 = (float*)weights_conv_1d_l3_start;
float* weights_conv_1d_bias_l3 = (float*)weights_conv_1d_bias_l3_start;
float* weights_bn_1d_l3 = (float*)weights_bn_1d_l3_start;

float* weights_conv_1d_l4 = (float*)weights_conv_1d_l4_start;
float* weights_conv_1d_bias_l4 = (float*)weights_conv_1d_bias_l4_start;
float* weights_bn_1d_l4 = (float*)weights_bn_1d_l4_start;

float* linear_mat_1 = (float*)linear_mat_1_start;
float* linear_add_1 = (float*)linear_add_1_start;

float* linear_mat_2 = (float*)linear_mat_2_start;
float* linear_add_2 = (float*)linear_add_2_start;

#endif
