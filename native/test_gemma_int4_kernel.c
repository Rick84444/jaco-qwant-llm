#include "gemma_int4_kernel.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static uint8_t packed[64];
static void set_weight(int idx, int val) {
  uint8_t v=(uint8_t)(val & 15);
  if (idx & 1)packed[idx/2]=(packed[idx/2]&15)|(uint8_t)(v<<4);
  else packed[idx/2]=(packed[idx/2]&240)|v;
}
int main(void) {
  /* 2 outputs, K=33: second output starts in the HIGH nibble.
   * Checks real packed-flattened row semantics and signed values. */
  const int K=33,N=2;
  int8_t x[33];
  float s[2]={.25f,.5f}, bias[2]={1.f,-2.f}, out[2]={0,0};
  int32_t wzp[2]={0,-1};
  int64_t ref[2]={0,0};
  for(int k=0;k<K;k++) x[k]=(int8_t)((k%11)-6);
  for(int row=0;row<N;row++)for(int k=0;k<K;k++){
    int w=(k+row*3)%16-8;
    set_weight(row*K+k,w);
    ref[row]+=(int64_t)(x[k]+2)*(w-wzp[row]);
  }
  uint64_t ops=0,skip=0,neon=0;
  JacoGemmaInt4Params p={0};
  p.input=x;p.input_len=K;p.input_scale=.125f;
  p.input_zero_point=-2;p.weights=packed;p.weights_len=(K*N+1)/2;
  p.k=K;p.n=N;p.weight_scales=s;p.weight_scales_len=N;
  p.weight_zero_points=wzp;p.weight_zero_points_len=N;
  p.bias=bias;p.bias_len=N;p.output=out;p.output_len=N;
  p.executed_products=&ops;p.skipped_products=&skip;p.neon_blocks=&neon;
  assert(jaco_gemma_int4_required_bytes(K,N)==33);
  assert(jaco_gemma_int4_gemv(&p)==0);
  for(int row=0;row<N;row++)
    assert(fabsf(out[row]-(float)((double)ref[row]*p.input_scale*s[row]+bias[row]))<1e-5f);
  assert(ops==K*N);
  p.skip_exact_zero=1;
  assert(jaco_gemma_int4_gemv(&p)==0);
  for(int row=0;row<N;row++)
    assert(fabsf(out[row]-(float)((double)ref[row]*p.input_scale*s[row]+bias[row]))<1e-5f);
  assert(ops+skip==K*N && skip>0);
  printf("Gemma INT8xINT4 C/NEON smoke OK (neon_blocks=%llu)\n",(unsigned long long)neon);
  return 0;
}
