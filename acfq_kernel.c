#define _POSIX_C_SOURCE 199309L /* clock_gettime and CLOCK_MONOTONIC under -std=c11 */
/*
 * acfq_kernel.c - activation-sparse GEMV over quantized weights.
 *
 * Problem: per-channel activation sparsity skips input rows, but fp32 rows
 * cost 4 bytes/weight. Dense Q4 already reads ~0.56 bytes/weight. The win
 * only exists when sparsity is applied on top of a quantized layout that
 * still lets a whole input row be skipped.
 *
 * Layout ("row-major quantized", K input rows, N outputs, N % 32 == 0):
 *   row i = [ G fp16 scales ][ quantized payload ],  G = N / 32
 *   Q8: payload N bytes, int8, w = s * q
 *   Q4: payload N/2 bytes, group of 32 packed in 16 bytes:
 *       byte k low nibble  -> weight k,      high nibble -> weight k + 16,
 *       w = s * (nibble - 8)
 * Skipping input channel i skips its entire row: scales and payload.
 *
 * Kernel: y[N] = sum over active i of x[i] * dequant(row i).
 * Per group the scalar a = x[i] * s is formed once, then 32 FMAs.
 * Four active rows are fused per pass so y is loaded/stored once per 4 rows.
 *
 * Formats: 0 = fp32 rows, 1 = Q8, 2 = Q4,
 *          3 = Q8 with fp16 arithmetic, 4 = Q4 with fp16 arithmetic.
 * 3/4 use the same bytes as 1/2. Dequant goes int8 -> int16 -> fp16 and up to
 * four rows are multiply-accumulated in fp16 before widening into the fp32
 * accumulator (roughly 0.4 instead of 0.9 instructions per weight). Needs
 * ARMv8.2 FP16 vector arithmetic; without it 3/4 fall back to 1/2.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

typedef _Float16 f16;

enum { FMT_F32 = 0, FMT_Q8 = 1, FMT_Q4 = 2, FMT_Q8H = 3, FMT_Q4H = 4 };

size_t acfq_row_bytes(int fmt, int N)
{
    int G = N / 32;
    if (fmt == FMT_F32) return (size_t)N * 4;
    if (fmt == FMT_Q8 || fmt == FMT_Q8H) return (size_t)G * 2 + (size_t)N;
    return (size_t)G * 2 + (size_t)N / 2;
}

/* ------------------------------------------------------------ fp32 rows */

static void f32_rows4(float *restrict y, const float *w0, const float *w1,
                      const float *w2, const float *w3,
                      float a0, float a1, float a2, float a3, int N)
{
    int j = 0;
#ifdef __ARM_NEON
    float32x4_t v0 = vdupq_n_f32(a0), v1 = vdupq_n_f32(a1);
    float32x4_t v2 = vdupq_n_f32(a2), v3 = vdupq_n_f32(a3);
    for (; j + 4 <= N; j += 4) {
        float32x4_t o = vld1q_f32(y + j);
        o = vfmaq_f32(o, v0, vld1q_f32(w0 + j));
        o = vfmaq_f32(o, v1, vld1q_f32(w1 + j));
        o = vfmaq_f32(o, v2, vld1q_f32(w2 + j));
        o = vfmaq_f32(o, v3, vld1q_f32(w3 + j));
        vst1q_f32(y + j, o);
    }
#endif
    for (; j < N; ++j) y[j] += a0 * w0[j] + a1 * w1[j] + a2 * w2[j] + a3 * w3[j];
}

static void f32_rows1(float *restrict y, const float *w, float a, int N)
{
    int j = 0;
#ifdef __ARM_NEON
    float32x4_t v = vdupq_n_f32(a);
    for (; j + 4 <= N; j += 4) vst1q_f32(y + j, vfmaq_f32(vld1q_f32(y + j), v, vld1q_f32(w + j)));
#endif
    for (; j < N; ++j) y[j] += a * w[j];
}

/* ------------------------------------------------------------ Q8 rows */

#ifdef __ARM_NEON
static inline void q8_fma16(float32x4_t acc[4], int8x16_t q, float32x4_t a)
{
    int16x8_t lo = vmovl_s8(vget_low_s8(q)), hi = vmovl_s8(vget_high_s8(q));
    acc[0] = vfmaq_f32(acc[0], a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
    acc[1] = vfmaq_f32(acc[1], a, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
    acc[2] = vfmaq_f32(acc[2], a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
    acc[3] = vfmaq_f32(acc[3], a, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
}
#endif

static void q8_rows(float *restrict y, const uint8_t *const *rows,
                    const float *xs, int nr, int N)
{
    int G = N / 32;
    for (int g = 0; g < G; ++g) {
        float *yg = y + g * 32;
#ifdef __ARM_NEON
        float32x4_t acc[8];
        for (int k = 0; k < 8; ++k) acc[k] = vld1q_f32(yg + 4 * k);
        for (int r = 0; r < nr; ++r) {
            const f16 *sc = (const f16 *)rows[r];
            const int8_t *q = (const int8_t *)(rows[r] + (size_t)G * 2) + g * 32;
            float32x4_t a = vdupq_n_f32(xs[r] * (float)sc[g]);
            q8_fma16(acc, vld1q_s8(q), a);
            q8_fma16(acc + 4, vld1q_s8(q + 16), a);
        }
        for (int k = 0; k < 8; ++k) vst1q_f32(yg + 4 * k, acc[k]);
#else
        float acc[32];
        memcpy(acc, yg, sizeof acc);
        for (int r = 0; r < nr; ++r) {
            const f16 *sc = (const f16 *)rows[r];
            const int8_t *q = (const int8_t *)(rows[r] + (size_t)G * 2) + g * 32;
            float a = xs[r] * (float)sc[g];
            for (int j = 0; j < 32; ++j) acc[j] += a * (float)q[j];
        }
        memcpy(yg, acc, sizeof acc);
#endif
    }
}

/* ------------------------------------------------------------ Q4 rows */

#ifdef __ARM_NEON
static inline void q4_fma32(float32x4_t acc[8], uint8x16_t b, float32x4_t a)
{
    int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
    int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), vdupq_n_s8(8));
    q8_fma16(acc, lo, a);
    q8_fma16(acc + 4, hi, a);
}
#endif

static void q4_rows(float *restrict y, const uint8_t *const *rows,
                    const float *xs, int nr, int N)
{
    int G = N / 32;
    for (int g = 0; g < G; ++g) {
        float *yg = y + g * 32;
#ifdef __ARM_NEON
        float32x4_t acc[8];
        for (int k = 0; k < 8; ++k) acc[k] = vld1q_f32(yg + 4 * k);
        for (int r = 0; r < nr; ++r) {
            const f16 *sc = (const f16 *)rows[r];
            const uint8_t *q = rows[r] + (size_t)G * 2 + g * 16;
            q4_fma32(acc, vld1q_u8(q), vdupq_n_f32(xs[r] * (float)sc[g]));
        }
        for (int k = 0; k < 8; ++k) vst1q_f32(yg + 4 * k, acc[k]);
#else
        float acc[32];
        memcpy(acc, yg, sizeof acc);
        for (int r = 0; r < nr; ++r) {
            const f16 *sc = (const f16 *)rows[r];
            const uint8_t *q = rows[r] + (size_t)G * 2 + g * 16;
            float a = xs[r] * (float)sc[g];
            for (int j = 0; j < 16; ++j) {
                acc[j] += a * (float)((int)(q[j] & 0x0F) - 8);
                acc[j + 16] += a * (float)((int)(q[j] >> 4) - 8);
            }
        }
        memcpy(yg, acc, sizeof acc);
#endif
    }
}


/* ------------------------------------------------------------ fp16 arithmetic */

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define ACFQ_HAVE_FP16 1

/* p[0..3] hold 32 outputs as fp16; add a * int8 weights (two 16-lane loads) */
static inline void h_fma32(float16x8_t p[4], int8x16_t q0, int8x16_t q1, float16x8_t a, int first)
{
    float16x8_t w0 = vcvtq_f16_s16(vmovl_s8(vget_low_s8(q0)));
    float16x8_t w1 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(q0)));
    float16x8_t w2 = vcvtq_f16_s16(vmovl_s8(vget_low_s8(q1)));
    float16x8_t w3 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(q1)));
    if (first) {
        p[0] = vmulq_f16(a, w0); p[1] = vmulq_f16(a, w1);
        p[2] = vmulq_f16(a, w2); p[3] = vmulq_f16(a, w3);
    } else {
        p[0] = vfmaq_f16(p[0], a, w0); p[1] = vfmaq_f16(p[1], a, w1);
        p[2] = vfmaq_f16(p[2], a, w2); p[3] = vfmaq_f16(p[3], a, w3);
    }
}

static inline void h_widen_add(float *yg, const float16x8_t p[4])
{
    for (int k = 0; k < 4; ++k) {
        float32x4_t lo = vld1q_f32(yg + 8 * k), hi = vld1q_f32(yg + 8 * k + 4);
        vst1q_f32(yg + 8 * k, vaddq_f32(lo, vcvt_f32_f16(vget_low_f16(p[k]))));
        vst1q_f32(yg + 8 * k + 4, vaddq_f32(hi, vcvt_high_f32_f16(p[k])));
    }
}

static inline float16x8_t h_scalar(float v)
{
    v = v > 65000.0f ? 65000.0f : (v < -65000.0f ? -65000.0f : v);
    return vdupq_n_f16((float16_t)v);
}

static void q8h_rows(float *restrict y, const uint8_t *const *rows,
                     const float *xs, int nr, int N)
{
    int G = N / 32;
    for (int g = 0; g < G; ++g) {
        float16x8_t p[4];
        for (int r = 0; r < nr; ++r) {
            const f16 *sc = (const f16 *)rows[r];
            const int8_t *q = (const int8_t *)(rows[r] + (size_t)G * 2) + g * 32;
            h_fma32(p, vld1q_s8(q), vld1q_s8(q + 16), h_scalar(xs[r] * (float)sc[g]), r == 0);
        }
        h_widen_add(y + g * 32, p);
    }
}

static void q4h_rows(float *restrict y, const uint8_t *const *rows,
                     const float *xs, int nr, int N)
{
    int G = N / 32;
    const uint8x16_t m = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    for (int g = 0; g < G; ++g) {
        float16x8_t p[4];
        for (int r = 0; r < nr; ++r) {
            const f16 *sc = (const f16 *)rows[r];
            uint8x16_t b = vld1q_u8(rows[r] + (size_t)G * 2 + g * 16);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, m)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), eight);
            h_fma32(p, lo, hi, h_scalar(xs[r] * (float)sc[g]), r == 0);
        }
        h_widen_add(y + g * 32, p);
    }
}
#else
#define ACFQ_HAVE_FP16 0
#endif

int acfq_has_fp16(void) { return ACFQ_HAVE_FP16; }

static void rows_dispatch(int fmt, float *y, const uint8_t *const *r, const float *xs, int nr, int N)
{
    switch (fmt) {
    case FMT_F32:
        if (nr == 4)
            f32_rows4(y, (const float *)r[0], (const float *)r[1], (const float *)r[2],
                      (const float *)r[3], xs[0], xs[1], xs[2], xs[3], N);
        else
            f32_rows1(y, (const float *)r[0], xs[0], N);
        break;
#if ACFQ_HAVE_FP16
    case FMT_Q8H: q8h_rows(y, r, xs, nr, N); break;
    case FMT_Q4H: q4h_rows(y, r, xs, nr, N); break;
#else
    case FMT_Q8H: q8_rows(y, r, xs, nr, N); break;
    case FMT_Q4H: q4_rows(y, r, xs, nr, N); break;
#endif
    case FMT_Q8: q8_rows(y, r, xs, nr, N); break;
    default: q4_rows(y, r, xs, nr, N); break;
    }
}

/* ------------------------------------------------------------ public */

/* y = sum_{t < n_idx} x[idx[t]] * row(idx[t]); idx == NULL means all K rows. */
void acfq_gemv(int fmt, const uint8_t *W, int K, int N, const float *x,
               const int *idx, int n_idx, float *y)
{
    size_t rb = acfq_row_bytes(fmt, N);
    int n = idx ? n_idx : K;
    memset(y, 0, sizeof(float) * (size_t)N);
    int t = 0;
    for (; t + 4 <= n; t += 4) {
        int i0 = idx ? idx[t] : t,     i1 = idx ? idx[t + 1] : t + 1;
        int i2 = idx ? idx[t + 2] : t + 2, i3 = idx ? idx[t + 3] : t + 3;
        const uint8_t *r[4] = { W + i0 * rb, W + i1 * rb, W + i2 * rb, W + i3 * rb };
        float xs[4] = { x[i0], x[i1], x[i2], x[i3] };
        rows_dispatch(fmt, y, r, xs, 4, N);
    }
    for (; t < n; ++t) {
        int i = idx ? idx[t] : t;
        const uint8_t *r[1] = { W + i * rb };
        float xs[1] = { x[i] };
        rows_dispatch(fmt, y, r, xs, 1, N);
    }
}

/* keep channel i when |x_i| > tau; returns count, indices ascending */
int acfq_mask(const float *x, int K, float tau, int *idx)
{
    int n = 0;
    for (int i = 0; i < K; ++i)
        if (fabsf(x[i]) > tau) idx[n++] = i;
    return n;
}

/* one token: tau < 0 means dense (no mask, all rows) */
int acfq_token(int fmt, const uint8_t *W, int K, int N, const float *x,
               float tau, int *idx, float *y)
{
    if (tau < 0.0f) {
        acfq_gemv(fmt, W, K, N, x, NULL, K, y);
        return K;
    }
    int n = acfq_mask(x, K, tau, idx);
    acfq_gemv(fmt, W, K, N, x, idx, n, y);
    return n;
}

/* all tokens of X (T x K) into Y (T x N); returns total active rows */
long acfq_run(int fmt, const uint8_t *W, int K, int N, const float *X, int T,
              float tau, int *idx, float *Y)
{
    long total = 0;
    for (int t = 0; t < T; ++t)
        total += acfq_token(fmt, W, K, N, X + (size_t)t * K, tau, idx, Y + (size_t)t * N);
    return total;
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* mean ns per token over reps passes of T tokens (y is N scratch) */
double acfq_bench(int fmt, const uint8_t *W, int K, int N, const float *X, int T,
                  float tau, int reps, int *idx, float *y)
{
    double t0 = now_s();
    for (int r = 0; r < reps; ++r)
        for (int t = 0; t < T; ++t)
            acfq_token(fmt, W, K, N, X + (size_t)t * K, tau, idx, y);
    return (now_s() - t0) * 1e9 / ((double)reps * T);
}

/* run for at least `seconds`, cycling tokens; returns tokens processed,
 * elapsed seconds in *elapsed. Used for energy sampling. */
long acfq_run_for(int fmt, const uint8_t *W, int K, int N, const float *X, int T,
                  float tau, double seconds, int *idx, float *y, double *elapsed)
{
    double t0 = now_s(), t1 = t0;
    long done = 0;
    while (t1 - t0 < seconds) {
        for (int t = 0; t < T; ++t)
            acfq_token(fmt, W, K, N, X + (size_t)t * K, tau, idx, y);
        done += T;
        t1 = now_s();
    }
    *elapsed = t1 - t0;
    return done;
}
