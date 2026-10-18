/*
 * acf_kernel.c - activation-sparse GEMV kernels, pure functions, no global state.
 *
 * Layout: W is row-major K x N (input channel i owns row W[i*N .. i*N+N-1]),
 * so skipping input channel i skips reading one full weight row.
 * y = x^T W  (x has K entries, y has N entries)
 */
#include <math.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static void axpy_row(float *restrict y, const float *restrict w, float a, int n)
{
    int j = 0;
#ifdef __ARM_NEON
    float32x4_t av = vdupq_n_f32(a);
    for (; j + 16 <= n; j += 16) {
        float32x4_t y0 = vld1q_f32(y + j),      y1 = vld1q_f32(y + j + 4);
        float32x4_t y2 = vld1q_f32(y + j + 8),  y3 = vld1q_f32(y + j + 12);
        y0 = vfmaq_f32(y0, av, vld1q_f32(w + j));
        y1 = vfmaq_f32(y1, av, vld1q_f32(w + j + 4));
        y2 = vfmaq_f32(y2, av, vld1q_f32(w + j + 8));
        y3 = vfmaq_f32(y3, av, vld1q_f32(w + j + 12));
        vst1q_f32(y + j, y0);      vst1q_f32(y + j + 4, y1);
        vst1q_f32(y + j + 8, y2);  vst1q_f32(y + j + 12, y3);
    }
#endif
    for (; j < n; ++j) y[j] += a * w[j];
}

/* Dense reference: every row read. Same inner loop as the sparse path, so the
 * speed comparison measures only the effect of skipping rows. */
void acf_gemv_dense(const float *W, int K, int N, const float *x, float *y)
{
    memset(y, 0, sizeof(float) * (size_t)N);
    for (int i = 0; i < K; ++i) {
        float a = x[i];
        if (a != 0.0f) axpy_row(y, W + (size_t)i * N, a, N);
    }
}

/* Sparse: only rows listed in idx[0..n_idx-1] are read. */
void acf_gemv_rows(const float *W, int K, int N, const float *x,
                   const int *idx, int n_idx, float *y)
{
    (void)K;
    memset(y, 0, sizeof(float) * (size_t)N);
    for (int t = 0; t < n_idx; ++t) {
        int i = idx[t];
        axpy_row(y, W + (size_t)i * N, x[i], N);
    }
}

/* Per-channel magnitude threshold: keep channel i if |x_i| > tau.
 * Returns number of kept channels; indices written ascending to idx. */
int acf_mask_threshold(const float *x, int K, float tau, int *idx)
{
    int n = 0;
    for (int i = 0; i < K; ++i) {
        float a = x[i];
        if (isfinite(a) && fabsf(a) > tau) idx[n++] = i;
    }
    return n;
}

/* Block energy mask, the method from the original ACFLayerBackend:
 * rank blocks of size B by sum(x^2), keep the highest until the kept energy
 * reaches target * total. With dynamic != 0 the target is raised by the
 * normalized Shannon entropy of the block energy distribution:
 *   target = base + (max_target - base) * H_norm^2
 * scratch_e must hold M floats and scratch_o M ints, M = ceil(K / B).
 * Output: channel indices (ascending) of every kept block. Returns count.
 * entropy_out / target_out may be NULL. */
int acf_mask_energy_blocks(const float *x, int K, int B,
                           float base, float max_target, int dynamic,
                           float *scratch_e, int *scratch_o,
                           int *idx, float *entropy_out, float *target_out)
{
    int M = (K + B - 1) / B;
    float total = 0.0f;
    for (int b = 0; b < M; ++b) {
        int s = b * B, e = s + B < K ? s + B : K;
        float acc = 0.0f;
        for (int i = s; i < e; ++i) {
            float v = isfinite(x[i]) ? x[i] : 0.0f;
            acc += v * v;
        }
        scratch_e[b] = acc;
        scratch_o[b] = b;
        total += acc;
    }
    if (total <= 1e-12f) total = 1.0f;

    float h_norm = 0.0f;
    if (M > 1) {
        float h = 0.0f;
        for (int b = 0; b < M; ++b) {
            float p = scratch_e[b] / total;
            if (p > 1e-7f) h -= p * logf(p);
        }
        h_norm = h / logf((float)M);
    }
    float target = dynamic ? base + (max_target - base) * h_norm * h_norm : base;

    /* insertion sort of block order by energy, descending; M is small */
    for (int a = 1; a < M; ++a) {
        int key = scratch_o[a];
        float ke = scratch_e[key];
        int c = a - 1;
        while (c >= 0 && scratch_e[scratch_o[c]] < ke) {
            scratch_o[c + 1] = scratch_o[c];
            --c;
        }
        scratch_o[c + 1] = key;
    }

    float goal = total * target, acc = 0.0f;
    int kept = 0;
    for (int r = 0; r < M; ++r) {
        acc += scratch_e[scratch_o[r]];
        ++kept;
        if (acc >= goal) break;
    }

    /* sort kept block ids ascending so rows are read in memory order */
    for (int a = 1; a < kept; ++a) {
        int key = scratch_o[a], c = a - 1;
        while (c >= 0 && scratch_o[c] > key) { scratch_o[c + 1] = scratch_o[c]; --c; }
        scratch_o[c + 1] = key;
    }

    int n = 0;
    for (int r = 0; r < kept; ++r) {
        int s = scratch_o[r] * B, e = s + B < K ? s + B : K;
        for (int i = s; i < e; ++i) idx[n++] = i;
    }
    if (entropy_out) *entropy_out = h_norm;
    if (target_out) *target_out = target;
    return n;
}

/* ---- timing loops (in C, so call overhead does not distort per-token times) ---- */
#include <time.h>

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* X is T x K row-major, Y is N scratch. Returns mean ns per token. */
double acf_bench_dense(const float *W, int K, int N, const float *X, int T,
                       int reps, float *Y)
{
    double t0 = now_ns();
    for (int r = 0; r < reps; ++r)
        for (int t = 0; t < T; ++t)
            acf_gemv_dense(W, K, N, X + (size_t)t * K, Y);
    return (now_ns() - t0) / ((double)reps * T);
}

/* Mask + sparse GEMV per token, mask recomputed every token. */
double acf_bench_threshold(const float *W, int K, int N, const float *X, int T,
                           float tau, int reps, float *Y, int *idx)
{
    double t0 = now_ns();
    for (int r = 0; r < reps; ++r)
        for (int t = 0; t < T; ++t) {
            const float *x = X + (size_t)t * K;
            int n = acf_mask_threshold(x, K, tau, idx);
            acf_gemv_rows(W, K, N, x, idx, n, Y);
        }
    return (now_ns() - t0) / ((double)reps * T);
}

/* Block-energy mask + sparse GEMV; mask recomputed every `interval` tokens
 * (interval = 1 means per token, 32 reproduces the original backend). */
double acf_bench_energy(const float *W, int K, int N, const float *X, int T,
                        int B, float base, float max_target, int dynamic,
                        int interval, int reps, float *Y, int *idx,
                        float *scratch_e, int *scratch_o)
{
    double t0 = now_ns();
    for (int r = 0; r < reps; ++r) {
        int n = 0;
        for (int t = 0; t < T; ++t) {
            const float *x = X + (size_t)t * K;
            if (t % interval == 0)
                n = acf_mask_energy_blocks(x, K, B, base, max_target, dynamic,
                                           scratch_e, scratch_o, idx, NULL, NULL);
            acf_gemv_rows(W, K, N, x, idx, n, Y);
        }
    }
    return (now_ns() - t0) / ((double)reps * T);
}
