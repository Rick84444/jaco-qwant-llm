#include "gemma_int4_kernel.h"
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <float.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define JACO_GEMMA_HAVE_NEON 1
#else
#define JACO_GEMMA_HAVE_NEON 0
#endif

size_t jaco_gemma_int4_required_bytes(int k, int n) {
  if (k <= 0 || n <= 0 || (size_t)k > SIZE_MAX / (size_t)n)
    return 0;
  const size_t elements = (size_t)k * (size_t)n;
  return elements / 2 + elements % 2;
}

static inline int unpack_signed_int4(const uint8_t *packed, size_t index) {
  const uint8_t byte = packed[index >> 1];
  const int nibble = (index & 1) ? (byte >> 4) : (byte & 15);
  return (nibble & 8) ? nibble - 16 : nibble;
}

#if JACO_GEMMA_HAVE_NEON
/* Sign-extend low and high nibbles, zip back into consecutive values,
 * then widen x and w before subtraction (input differences need 9 bits).
 * A 32-element partial sum fits int32; total is accumulated in int64.
 */
static inline int32_t dot8(int8x8_t a, int8x8_t w, int8_t azp, int8_t wzp) {
  const int16x8_t ax = vsubl_s8(a, vdup_n_s8(azp));
  const int16x8_t wx = vsubl_s8(w, vdup_n_s8(wzp));
  const int32x4_t lo = vmull_s16(vget_low_s16(ax), vget_low_s16(wx));
  const int32x4_t hi = vmull_s16(vget_high_s16(ax), vget_high_s16(wx));
  return vaddvq_s32(vaddq_s32(lo, hi));
}
static inline int64_t dot32_neon(const int8_t *x, const uint8_t *packed,
                                  int8_t azp, int8_t wzp) {
  const uint8x16_t bytes = vld1q_u8(packed);
  const int8x16_t lows = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(bytes, 4)),4);
  const int8x16_t highs = vshrq_n_s8(vreinterpretq_s8_u8(bytes),4);
  const int8x16x2_t w = vzipq_s8(lows, highs);
  const int8x16_t a0 = vld1q_s8(x);
  const int8x16_t a1 = vld1q_s8(x+16);
  int64_t sum = 0;
  sum += dot8(vget_low_s8(a0),vget_low_s8(w.val[0]),azp,wzp);
  sum += dot8(vget_high_s8(a0),vget_high_s8(w.val[0]),azp,wzp);
  sum += dot8(vget_low_s8(a1),vget_low_s8(w.val[1]),azp,wzp);
  sum += dot8(vget_high_s8(a1),vget_high_s8(w.val[1]),azp,wzp);
  return sum;
}
#endif

int jaco_gemma_int4_gemv(const JacoGemmaInt4Params *p) {
  if (!p || !p->input || !p->weights || !p->weight_scales || !p->output ||
      p->k <= 0 || p->n <= 0 || p->input_zero_point < -128 ||
      p->input_zero_point > 127 || !isfinite(p->input_scale) ||
      p->input_scale <= 0 || (p->skip_exact_zero != 0 && p->skip_exact_zero != 1))
    return JACO_GEMMA_INT4_INVALID;
  const size_t bytes = jaco_gemma_int4_required_bytes(p->k,p->n);
  if (!bytes) return JACO_GEMMA_INT4_INVALID;
  if (p->input_len < (size_t)p->k || p->weights_len < bytes ||
      p->weight_scales_len < (size_t)p->n || p->output_len < (size_t)p->n ||
      (p->weight_zero_points && p->weight_zero_points_len < (size_t)p->n) ||
      (p->bias && p->bias_len < (size_t)p->n))
    return JACO_GEMMA_INT4_BOUNDS;
  /* Validate all metadata before writing any outputs. */
  for (int row = 0; row < p->n; ++row) {
    const float scale = p->weight_scales[row];
    const int32_t wzp = p->weight_zero_points ? p->weight_zero_points[row] : 0;
    if (!isfinite(scale) || scale <= 0 || wzp < -8 || wzp > 7 ||
        (p->bias && !isfinite(p->bias[row])))
      return JACO_GEMMA_INT4_INVALID;
  }
  uint64_t done=0, skipped=0, blocks=0;
  for (int row = 0; row < p->n; ++row) {
    const int32_t wzp = p->weight_zero_points ? p->weight_zero_points[row] : 0;
    const size_t row_start = (size_t)row * (size_t)p->k;
    int64_t accum=0;
    int col=0;
#if JACO_GEMMA_HAVE_NEON
    if (!p->skip_exact_zero && !(row_start & 1)) {
      for (; col+32 <= p->k; col+=32) {
        accum+=dot32_neon(p->input+col,
                   p->weights+((row_start+(size_t)col)/2),
                   (int8_t)p->input_zero_point,(int8_t)wzp);
        done+=32;
        ++blocks;
      }
    }
#endif
    for (;col<p->k;++col) {
      const int32_t a=(int32_t)p->input[col]-p->input_zero_point;
      if (p->skip_exact_zero && a==0) {
        ++skipped;
        continue;
      }
      const int32_t w=unpack_signed_int4(p->weights,row_start+(size_t)col)-wzp;
      accum+=(int64_t)a*(int64_t)w;
      ++done;
    }
    const double value=(double)accum*(double)p->input_scale *
           (double)p->weight_scales[row] + (p->bias ? (double)p->bias[row] : 0);
    if (!isfinite(value) || value > (double)FLT_MAX || value < -(double)FLT_MAX)
      return JACO_GEMMA_INT4_INVALID;
    p->output[row]=(float)value;
  }
  if (p->executed_products) *p->executed_products=done;
  if (p->skipped_products) *p->skipped_products=skipped;
  if (p->neon_blocks) *p->neon_blocks=blocks;
  return JACO_GEMMA_INT4_OK;
}
