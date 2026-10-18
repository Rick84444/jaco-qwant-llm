#ifndef JACO_ACFQ_BACKEND_H
#define JACO_ACFQ_BACKEND_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental native operator ABI. NOT a LiteRT delegate. */
enum { JACO_ACFQ_F32=0, JACO_ACFQ_Q8=1, JACO_ACFQ_Q4=2 };
enum { JACO_ACFQ_OK=0, JACO_ACFQ_EINVAL=-1, JACO_ACFQ_EBOUNDS=-2 };
size_t jaco_acfq_packed_row_bytes(int fmt, int n);
int jaco_acfq_execute(int fmt, const void *weights, size_t weights_bytes,
                      int k, int n, const float *input, size_t input_len,
                      const int *indices, size_t count,
                      float *output, size_t output_len);
#ifdef __cplusplus
}
#endif
#endif
