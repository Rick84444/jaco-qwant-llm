#ifndef JACO_GEMMA_INT4_KERNEL_H
#define JACO_GEMMA_INT4_KERNEL_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/* Experimental TFLite-compatible integer projection math, not a LiteRT delegate.
 * Weights are signed two's-complement INT4, flattened row-major [N,K],
 * two weights per byte, low nibble first, with no row padding.
 * Output is dequantized FP32, not an INT8 FullyConnected output.
 */
enum {
  JACO_GEMMA_INT4_OK=0,
  JACO_GEMMA_INT4_INVALID=-1,
  JACO_GEMMA_INT4_BOUNDS=-2
};
typedef struct {
  const int8_t *input;       /* K quantized activations */
  size_t input_len;
  float input_scale;         /* per-tensor, positive */
  int32_t input_zero_point;
  const uint8_t *weights;   /* packed N*K signed 4-bit weights */
  size_t weights_len;
  int k, n;
  const float *weight_scales;      /* N per-channel scales */
  size_t weight_scales_len;
  const int32_t *weight_zero_points; /* optional N zero points, NULL means zero */
  size_t weight_zero_points_len;
  const float *bias;          /* optional dequantized bias, NULL means zero */
  size_t bias_len;
  float *output;              /* N dequantized outputs */
  size_t output_len;
  int skip_exact_zero;        /* only skip input == input_zero_point */
  uint64_t *executed_products;  /* optional diagnostic counters */
  uint64_t *skipped_products;
  uint64_t *neon_blocks;
} JacoGemmaInt4Params;
size_t jaco_gemma_int4_required_bytes(int k, int n);
int jaco_gemma_int4_gemv(const JacoGemmaInt4Params *params);
#ifdef __cplusplus
}
#endif
#endif
