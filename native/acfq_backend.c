/* Safe native operator boundary for ACFQ, using an existing kernel. */
#include "acfq_backend.h"
#include <stdint.h>
#include <limits.h>
#include <stddef.h>
#include <math.h>
/* Provided by acfq_kernel.c */
extern void acfq_gemv(int fmt, const uint8_t *w, int k, int n,
                      const float *x, const int *idx, int nidx, float *y);
size_t jaco_acfq_packed_row_bytes(int fmt, int n) {
  if (n <= 0 || n % 32 != 0) return 0;
  if (fmt == JACO_ACFQ_F32) return (size_t)n * sizeof(float);
  if (fmt == JACO_ACFQ_Q8) return (size_t)n + ((size_t)n / 32) * 2;
  if (fmt == JACO_ACFQ_Q4) return (size_t)n / 2 + ((size_t)n / 32) * 2;
  return 0;
}
int jaco_acfq_execute(int fmt, const void *weights, size_t weights_bytes,
                      int k, int n, const float *input, size_t input_len,
                      const int *indices, size_t count,
                      float *output, size_t output_len) {
  size_t row=jaco_acfq_packed_row_bytes(fmt,n);
  if (!row || k<=0 || !weights || !input || !output || input_len<(size_t)k ||
      output_len<(size_t)n || (indices==NULL && count!=0) ||
      (indices && count>(size_t)k) || count>(size_t)INT_MAX)
    return JACO_ACFQ_EINVAL;
  if ((size_t)k>SIZE_MAX/row || weights_bytes < (size_t)k*row) return JACO_ACFQ_EBOUNDS;
  /* The underlying kernel assumes aligned F32 weights; caller must provide it. */
  if (fmt==JACO_ACFQ_F32 && (uintptr_t)weights % _Alignof(float)) return JACO_ACFQ_EINVAL;
  if (indices) {
    int last=-1;
    for (size_t i=0;i<count;++i) {
      if (indices[i]<=last || indices[i]>=k) return JACO_ACFQ_EINVAL;
      last=indices[i];
    }
  }
  for (int i=0;i<k;++i) if (!isfinite(input[i])) return JACO_ACFQ_EINVAL;
  acfq_gemv(fmt,(const uint8_t*)weights,k,n,input,indices,(int)count,output);
  return JACO_ACFQ_OK;
}
