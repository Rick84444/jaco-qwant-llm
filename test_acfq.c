/*
 * test_acfq.c - checks acfq_kernel.c against a float64 reference for every
 * format (fp32, Q8, Q4), dense and sparse index lists, odd row counts
 * (exercises the 4-row fused path and the 1-row tail).
 * Build: cc -O2 test_acfq.c -o test_acfq -lm && ./test_acfq
 */
#include "acfq_kernel.c"
#include <stdio.h>
#include <stdlib.h>

static uint32_t rng_state = 12345u;

static uint32_t rnd(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float rndf(void) { return (float)(rnd() & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f; }

static double deq(int fmt, const uint8_t *W, int N, int i, int j)
{
    size_t rb = acfq_row_bytes(fmt, N);
    const uint8_t *row = W + (size_t)i * rb;
    if (fmt == FMT_F32) return ((const float *)row)[j];
    int G = N / 32, g = j / 32, k = j % 32;
    double s = (double)((const f16 *)row)[g];
    if (fmt == FMT_Q8 || fmt == FMT_Q8H) return s * (double)((const int8_t *)(row + G * 2))[j];
    uint8_t b = (row + G * 2 + g * 16)[k % 16];
    int q = k < 16 ? (b & 0x0F) : (b >> 4);
    return s * (double)(q - 8);
}

static int check(int fmt, int K, int N, const int *idx, int n_idx)
{
    size_t rb = acfq_row_bytes(fmt, N);
    uint8_t *W = malloc(rb * K);
    int G = N / 32;
    for (int i = 0; i < K; ++i) {
        uint8_t *row = W + (size_t)i * rb;
        if (fmt == FMT_F32) {
            for (int j = 0; j < N; ++j) ((float *)row)[j] = rndf();
        } else {
            for (int g = 0; g < G; ++g) ((f16 *)row)[g] = (f16)(0.01f + 0.05f * (rndf() + 1.0f));
            size_t pb = (fmt == FMT_Q8 || fmt == FMT_Q8H) ? (size_t)N : (size_t)N / 2;
            for (size_t b = 0; b < pb; ++b) row[G * 2 + b] = (uint8_t)rnd();
        }
    }
    float *x = malloc(sizeof(float) * K), *y = malloc(sizeof(float) * N);
    for (int i = 0; i < K; ++i) x[i] = rndf();
    acfq_gemv(fmt, W, K, N, x, idx, n_idx, y);
    int n = idx ? n_idx : K;
    double num = 0, den = 0;
    for (int j = 0; j < N; ++j) {
        double ref = 0;
        for (int t = 0; t < n; ++t) {
            int i = idx ? idx[t] : t;
            ref += (double)x[i] * deq(fmt, W, N, i, j);
        }
        num += (ref - y[j]) * (ref - y[j]);
        den += ref * ref;
    }
    double rel = sqrt(num / (den + 1e-30));
    double tol = (fmt >= FMT_Q8H && acfq_has_fp16()) ? 2e-3 : 1e-5;
    int ok = rel < tol;
    printf("fmt=%d K=%d N=%d rows=%d  rel_err=%.2e  %s\n", fmt, K, N, n, rel, ok ? "PASS" : "FAIL");
    free(W); free(x); free(y);
    return ok;
}

int main(void)
{
#ifdef __ARM_NEON
    printf("path: NEON, fp16 arithmetic %s\n", acfq_has_fp16() ? "on" : "off (3/4 use 1/2)");
#else
    printf("path: scalar\n");
#endif
    int all = 1;
    int sparse[37];
    for (int t = 0; t < 37; ++t) sparse[t] = t * 3 + (t & 1);
    for (int fmt = 0; fmt < 5; ++fmt) {
        all &= check(fmt, 131, 96, NULL, 0);
        all &= check(fmt, 131, 96, sparse, 37);
        all &= check(fmt, 7, 32, NULL, 0);
        all &= check(fmt, 512, 256, sparse, 37);
    }
    printf("%s\n", all ? "ALL PASS" : "FAILURES");
    return all ? 0 : 1;
}
