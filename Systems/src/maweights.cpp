#include <cstddef>

// Store all of the weights as binaries and link them

extern const unsigned char weights_conv_1d_l1_start[];
extern const unsigned char weights_conv_1d_bias_l1_start[];
extern const unsigned char weights_bn_1d_l1_start[];

float* weights_conv_1d_l1      = (float*)weights_conv_1d_l1_start;
float* weights_conv_1d_bias_l1 = (float*)weights_conv_1d_bias_l1_start;
float* weights_bn_1d_l1        = (float*)weights_bn_1d_l1_start;



extern const unsigned char weights_conv_1d_l2_start[];
extern const unsigned char weights_conv_1d_bias_l2_start[];
extern const unsigned char weights_bn_1d_l2_start[];

float* weights_conv_1d_l2      = (float*)weights_conv_1d_l2_start;
float* weights_conv_1d_bias_l2 = (float*)weights_conv_1d_bias_l2_start;
float* weights_bn_1d_l2        = (float*)weights_bn_1d_l2_start;



extern const unsigned char weights_conv_1d_l3_start[];
extern const unsigned char weights_conv_1d_bias_l3_start[];
extern const unsigned char weights_bn_1d_l3_start[];

float* weights_conv_1d_l3      = (float*)weights_conv_1d_l3_start;
float* weights_conv_1d_bias_l3 = (float*)weights_conv_1d_bias_l3_start;
float* weights_bn_1d_l3        = (float*)weights_bn_1d_l3_start;



extern const unsigned char weights_conv_1d_l4_start[];
extern const unsigned char weights_conv_1d_bias_l4_start[];
extern const unsigned char weights_bn_1d_l4_start[];

float* weights_conv_1d_l4      = (float*)weights_conv_1d_l4_start;
float* weights_conv_1d_bias_l4 = (float*)weights_conv_1d_bias_l4_start;
float* weights_bn_1d_l4        = (float*)weights_bn_1d_l4_start;



extern const unsigned char linear_mat_1_start[];
extern const unsigned char linear_add_1_start[];

float* linear_mat_1 = (float*)linear_mat_1_start;
float* linear_add_1 = (float*)linear_add_1_start;



extern const unsigned char linear_mat_2_start[];
extern const unsigned char linear_add_2_start[];

float* linear_mat_2 = (float*)linear_mat_2_start;
float* linear_add_2 = (float*)linear_add_2_start;