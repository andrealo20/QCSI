#include "qcsi/features.h"

/**
 * Integer square root of a 64-bit value, by binary digit-by-digit descent.
 *
 * Used instead of sqrt() from libm because the whole processing path is
 * meant to run without floating point: pulling in the soft-float library for
 * one square root would be self-defeating on a target with no FPU. The loop
 * runs a fixed number of iterations and uses only shifts, adds and compares.
 */
static uint32_t isqrt64(uint64_t v)
{
    uint64_t rem = 0u, root = 0u;
    int i;

    for (i = 0; i < 32; ++i) {
        root <<= 1;
        rem = (rem << 2) | (v >> 62);
        v <<= 2;
        if (root < rem) {
            rem -= root | 1u;
            root += 2u;
        }
    }
    return (uint32_t)(root >> 1);
}

/* ------------------------------------------------------------------ */
/* Amplitude statistics                                                */
/* ------------------------------------------------------------------ */

qdsp_status_t qcsi_amplitude_stats(const q15_t *window, size_t n_frames,
                                   size_t n_sub, q15_t *out)
{
    size_t k, f;

    if (window == NULL || out == NULL || n_frames == 0u || n_sub == 0u) {
        return QDSP_ERR_ARG;
    }

    for (k = 0u; k < n_sub; ++k) {
        q63_t sum = 0;
        q63_t sum_sq = 0;
        q15_t lo = QDSP_Q15_MAX;
        q15_t hi = QDSP_Q15_MIN;
        int32_t mean;
        q63_t var;

        for (f = 0u; f < n_frames; ++f) {
            q15_t x = window[f * n_sub + k];
            sum += (q63_t)x;
            /* Accumulating squares in 64 bits: n_frames * 2^30 would
               overflow 32 bits for any realistic window length. */
            sum_sq += (q63_t)x * (q63_t)x;
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }

        mean = (int32_t)(sum / (q63_t)n_frames);

        /* Variance as (n*sum_sq - sum^2) / n^2, not as E[x^2] - mean^2.
           The two are algebraically identical but not numerically: the
           second form squares an integer-truncated mean, and that truncation
           error is multiplied by the mean itself. On a CSI amplitude — a
           small ripple sitting on a large static component — the inflation
           is larger than the quantity being measured. Measured on a 1%
           ripple at 0.9 mean, the truncated form overestimated the spread by
           10%; this one is within 1%.

           Range check: sum_sq <= n*2^30 and sum^2 <= n^2*2^30, so with a
           64-bit accumulator the window may be up to ~2^16 frames. */
        var = ((q63_t)n_frames * sum_sq - sum * sum) /
              ((q63_t)n_frames * (q63_t)n_frames);
        if (var < 0) var = 0;   /* only reachable through rounding */

        out[k * QCSI_STATS_PER_SUBCARRIER + QCSI_STAT_MEAN] =
            qdsp_sat_q15(mean);
        /* var is in Q30; its square root is therefore back in Q15, on the
           same scale as the signal, which is where the resolution is. */
        out[k * QCSI_STATS_PER_SUBCARRIER + QCSI_STAT_VAR] =
            qdsp_sat_q15((int32_t)isqrt64((uint64_t)var));
        out[k * QCSI_STATS_PER_SUBCARRIER + QCSI_STAT_PTP] =
            qdsp_sub_q15(hi, lo);
    }

    return QDSP_OK;
}

qdsp_status_t qcsi_remove_static_component(q15_t *window, size_t n_frames,
                                           size_t n_sub)
{
    size_t k, f;

    if (window == NULL || n_frames == 0u || n_sub == 0u) {
        return QDSP_ERR_ARG;
    }

    for (k = 0u; k < n_sub; ++k) {
        q63_t sum = 0;
        q15_t mean;

        for (f = 0u; f < n_frames; ++f) {
            sum += (q63_t)window[f * n_sub + k];
        }
        mean = qdsp_sat_q15((int32_t)(sum / (q63_t)n_frames));

        for (f = 0u; f < n_frames; ++f) {
            window[f * n_sub + k] = qdsp_sub_q15(window[f * n_sub + k], mean);
        }
    }

    return QDSP_OK;
}

/* ------------------------------------------------------------------ */
/* Doppler spectrum                                                    */
/* ------------------------------------------------------------------ */

qdsp_status_t qcsi_doppler_power(const q15_t *window, size_t n_frames,
                                 size_t n_sub, size_t sub, uint16_t n_fft,
                                 qdsp_cplx_q15 *scratch, q31_t *out)
{
    size_t f;
    uint16_t i;
    qdsp_status_t st;

    if (window == NULL || scratch == NULL || out == NULL ||
        n_frames == 0u || n_sub == 0u || sub >= n_sub ||
        !qdsp_fft_size_is_valid(n_fft) || (size_t)n_fft < n_frames) {
        return QDSP_ERR_ARG;
    }

    for (f = 0u; f < n_frames; ++f) {
        scratch[f].re = window[f * n_sub + sub];
        scratch[f].im = 0;
    }
    /* Zero padding interpolates the spectrum; it adds no information but
       makes the peak easier to locate. */
    for (i = (uint16_t)n_frames; i < n_fft; ++i) {
        scratch[i].re = 0;
        scratch[i].im = 0;
    }

    st = qdsp_fft_q15(scratch, n_fft);
    if (st != QDSP_OK) {
        return st;
    }

    /* Real input, so the second half mirrors the first: only n_fft/2 bins
       are written. Power is kept in Q31 because squaring two Q15 values
       fills 30 bits and clipping it back to Q15 would throw away the small
       bins, which are exactly the ones a classifier looks at. */
    for (i = 0u; i < (uint16_t)(n_fft / 2u); ++i) {
        int32_t re = (int32_t)scratch[i].re;
        int32_t im = (int32_t)scratch[i].im;
        out[i] = (q31_t)(re * re + im * im);
    }

    return QDSP_OK;
}

size_t qcsi_dominant_doppler_bin(const q31_t *power, size_t n_bins)
{
    size_t i, best = 0u;
    q31_t peak = 0;

    if (power == NULL || n_bins < 2u) {
        return 0u;
    }

    /* Bin 0 is skipped: it holds whatever static component survived
       qcsi_remove_static_component(), which is not motion. */
    for (i = 1u; i < n_bins; ++i) {
        if (power[i] > peak) {
            peak = power[i];
            best = i;
        }
    }

    return best;
}
