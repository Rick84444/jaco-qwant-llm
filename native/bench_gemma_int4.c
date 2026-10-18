/*
 * Bench the standalone JACO Gemma INT8xINT4 kernel on documented decoder
 * tensor dimensions. The reference is independent scalar integer GEMV.
 * No LiteRT kernels or Gemma tokens run here.
 */
#define _POSIX_C_SOURCE 200809L
#include "gemma_int4_kernel.h"
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile double checksum_sink=0.0;

static double now_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC,&ts)!=0) {
    perror("clock_gettime"); exit(2);
  }
  return (double)ts.tv_sec*1e9+(double)ts.tv_nsec;
}
static int cmp_double(const void *a,const void *b) {
  const double x=*(const double*)a, y=*(const double*)b;
  return (x>y)-(x<y);
}
static double median(double *v,int n) {
  qsort(v,(size_t)n,sizeof(*v),cmp_double);
  return n%2 ? v[n/2] : (v[n/2-1]+v[n/2])*.5;
}
static inline int int4(const uint8_t *w,size_t index) {
  int nib=(index&1) ? (w[index/2]>>4) : (w[index/2]&15);
  return (nib&8) ? nib-16 : nib;
}
static void scalar_reference(const JacoGemmaInt4Params *p) {
  for (int row=0;row<p->n;row++) {
    int64_t accum=0;
    const size_t base=(size_t)row*(size_t)p->k;
    const int32_t wzp=p->weight_zero_points ? p->weight_zero_points[row] : 0;
    for(int col=0;col<p->k;col++) {
      const int32_t a=(int32_t)p->input[col]-p->input_zero_point;
      const int32_t w=int4(p->weights,base+(size_t)col)-wzp;
      accum+=(int64_t)a*w;
    }
    const double bias=p->bias ? (double)p->bias[row] : 0;
    p->output[row]=(float)((double)accum*p->input_scale*p->weight_scales[row]+bias);
  }
}
static void *checked_calloc(size_t count,size_t size) {
  void *p=calloc(count,size);
  if (!p) { fprintf(stderr,"allocation failed\n"); exit(2); }
  return p;
}
static double measure(JacoGemmaInt4Params *p,int reference,int repeats) {
  double *times=(double*)checked_calloc((size_t)repeats,sizeof(double));
  for(int i=-2;i<repeats;i++) {
    double start=now_ns();
    if (reference) scalar_reference(p);
    else if(jaco_gemma_int4_gemv(p)!=JACO_GEMMA_INT4_OK) {
      fprintf(stderr,"native kernel failed during benchmark\n");exit(3);
    }
    const double elapsed=now_ns()-start;
    checksum_sink+=p->output[(i+32)%p->n];
    if(i>=0)times[i]=elapsed/1000.;
  }
  const double us=median(times,repeats);
  free(times);
  return us;
}
static double max_abs_error(const float *a,const float *b,int n) {
  double worst=0;
  for(int j=0;j<n;j++) {
    const double err=fabs((double)a[j]-b[j]);
    if(err>worst)worst=err;
    if(err>0.0002+0.00002*fabs((double)a[j])) {
      fprintf(stderr,"FAIL numerical difference at %d: ref=%g kernel=%g\n",
              j,(double)a[j],(double)b[j]);
      exit(4);
    }
  }
  return worst;
}
static void run_case(int k,int n,int zero_pct,int reps) {
  const size_t wb=jaco_gemma_int4_required_bytes(k,n);
  if(!wb) {fprintf(stderr,"invalid dimensions\n");exit(2);}
  uint8_t *weights=(uint8_t*)checked_calloc(wb,1);
  int8_t *x=(int8_t*)checked_calloc((size_t)k,sizeof(int8_t));
  float *scales=(float*)checked_calloc((size_t)n,sizeof(float));
  float *bias=(float*)checked_calloc((size_t)n,sizeof(float));
  float *oracle=(float*)checked_calloc((size_t)n,sizeof(float));
  float *dense=(float*)checked_calloc((size_t)n,sizeof(float));
  float *sparse=(float*)checked_calloc((size_t)n,sizeof(float));
  size_t zeros=0;
  /* Model-compatible shape and nominal format, SYNTHETIC values only. */
  for(size_t i=0;i<(size_t)k*(size_t)n;i++) {
    uint8_t nib=(uint8_t)((i*17u+(i>>6)+3u)&15u);
    if(i&1)weights[i/2]=(uint8_t)((weights[i/2]&15u)|(nib<<4));
    else weights[i/2]=(uint8_t)((weights[i/2]&240u)|nib);
  }
  for(int i=0;i<k;i++) {
    const int force_zero=((i*131u+7u)%100u)<(unsigned)zero_pct;
    x[i]=force_zero ? -3 : (int8_t)(((i*97u+(i/11u)+71u)%255u)-127);
    if(x[i]==-3)zeros++;
  }
  for(int j=0;j<n;j++) {
    scales[j]=0.0002f*(float)(1+j%23);
    bias[j]=0.01f*(float)(j%7);
  }
  uint64_t dense_products=0,sparse_products=0,skipped=0,neon_blocks=0;
  JacoGemmaInt4Params p={0};
  p.input=x;p.input_len=(size_t)k;
  p.input_scale=0.0075f;p.input_zero_point=-3;
  p.weights=weights;p.weights_len=wb;
  p.k=k;p.n=n;
  p.weight_scales=scales;p.weight_scales_len=(size_t)n;
  p.bias=bias;p.bias_len=(size_t)n;
  p.output_len=(size_t)n;
  p.output=oracle;scalar_reference(&p);
  p.output=dense;p.skip_exact_zero=0;
  p.executed_products=&dense_products;p.neon_blocks=&neon_blocks;
  if(jaco_gemma_int4_gemv(&p)) {fprintf(stderr,"dense failed\n");exit(3);}
  const double dense_error=max_abs_error(oracle,dense,n);
  const uint64_t dense_blocks=neon_blocks;
  p.neon_blocks=NULL; /* Preserve dense evidence; sparse is scalar. */
  p.output=sparse;p.skip_exact_zero=1;
  p.executed_products=&sparse_products;
  p.skipped_products=&skipped;
  if(jaco_gemma_int4_gemv(&p)) {fprintf(stderr,"sparse failed\n");exit(3);}
  const double sparse_error=max_abs_error(oracle,sparse,n);
  if(dense_products!=(uint64_t)k*(uint64_t)n ||
     sparse_products+skipped!=(uint64_t)k*(uint64_t)n ||
     skipped!=(uint64_t)zeros*(uint64_t)n) {
    fprintf(stderr,"FAIL operation count mismatch\n");exit(4);
  }
  p.output=oracle;const double scalar_us=measure(&p,1,reps);
  p.output=dense;p.skip_exact_zero=0;
  p.skipped_products=NULL;const double dense_us=measure(&p,0,reps);
  p.output=sparse;p.skip_exact_zero=1;
  p.skipped_products=&skipped;const double sparse_us=measure(&p,0,reps);
  printf("%d,%d,%d,%.2f,%.3f,%.3f,%.3f,%.4f,%.4f,%.8f,%.8f,%" PRIu64 ",%" PRIu64 "\n",
    k,n,zero_pct,100.0*(double)zeros/(double)k,
    scalar_us,dense_us,sparse_us,scalar_us/dense_us,
    scalar_us/sparse_us,dense_error,sparse_error,
    dense_blocks,skipped);
  fflush(stdout);
  free(weights);free(x);free(scales);free(bias);
  free(oracle);free(dense);free(sparse);
}
int main(int argc,char **argv) {
  int quick=0,reps=7;
  for(int i=1;i<argc;i++) {
    if(!strcmp(argv[i],"--quick")) quick=1;
    else if(!strcmp(argv[i],"--reps") && i+1<argc) {
      char *end=NULL;
      long value=strtol(argv[++i],&end,10);
      if(*end || value<1 || value>100) {
        fprintf(stderr,"--reps must be in [1,100]\n");return 2;
      }
      reps=(int)value;
    } else {fprintf(stderr,"Usage: %s [--quick] [--reps 1..100]\n",argv[0]);return 2;}
  }
  puts("# Synthetic INT8xINT4 data; scalar reference is NOT LiteRT; per-inference microbenchmark.");
  puts("K,N,target_zero_pct,actual_zero_pct,scalar_us,acfq_dense_us,acfq_exact_zero_us,dense_vs_scalar,exact_zero_vs_scalar,dense_abs_error,exact_zero_abs_error,dense_neon_blocks,exact_zero_skipped_products");
  if(quick)run_case(64,32,25,reps);
  else {
    const int widths[]={256,2048};
    const int zero_rates[]={0,25,50};
    for(size_t i=0;i<2;i++)for(size_t j=0;j<3;j++)
      run_case(1536,widths[i],zero_rates[j],reps);
  }
  if(!isfinite(checksum_sink))return 5;
  return 0;
}
