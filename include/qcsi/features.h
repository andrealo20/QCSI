/**
 * @file features.h
 * @brief Fixed-point feature extraction from a window of CSI frames.
 *
 * A single CSI frame says little. What carries the information is how the
 * channel evolves over a window of consecutive frames: a moving body
 * modulates each subcarrier over time, and the rate of that modulation is
 * the Doppler signature of the motion.
 *
 * Two families of features are computed here, deliberately kept simple:
 *
 *   - **Amplitude statistics** per subcarrier across the window (mean,
 *     variance, peak-to-peak). Cheap, robust, and what most published work
 *     relies on.
 *   - **Doppler spectrum**: an FFT along the time axis of one subcarrier,
 *     giving the distribution of motion rates. This is where qdsp's FFT does
 *     the work.
 *
 * There is no neural network here and that is a choice, not an omission. A
 * small feature set feeding a linear classifier is interpretable, fits in a
 * few kilobytes, and makes the cost of every stage visible. A quantised
 * network can be added later on top of the same features if the accuracy
 * gap justifies it.
 *
 * ## Memory model
 *
 * As in qdsp, nothing is allocated. The caller owns every buffer, and the
 * window is passed as a flat array in frame-major order:
 *
 *     window[f * n_sub + k]   frame f, subcarrier k
 *
 * which is the order data arrives in, so no transpose is needed on the
 * ingest path.
 */
#ifndef QCSI_FEATURES_H
#define QCSI_FEATURES_H

#include <stddef.h>
#include <stdint.h>

#include "qdsp/fft.h"
#include "qdsp/fixed.h"
#include "qdsp/status.h"

#include "qcsi/phase.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Number of statistics produced per subcarrier by qcsi_amplitude_stats(). */
#define QCSI_STATS_PER_SUBCARRIER 3

/** Indices into the per-subcarrier statistics block. */
typedef enum {
    QCSI_STAT_MEAN = 0,   /**< mean amplitude over the window, Q15 */
    QCSI_STAT_VAR  = 1,   /**< standard deviation, Q15 (not variance) */
    QCSI_STAT_PTP  = 2    /**< peak-to-peak spread, Q15 */
} qcsi_stat_t;

/* ------------------------------------------------------------------ */
/* Amplitude statistics                                                */
/* ------------------------------------------------------------------ */

/**
 * Mean, standard deviation and peak-to-peak amplitude per subcarrier.
 *
 * The spread is returned as a standard deviation rather than a variance,
 * and that is a correction rather than a preference. A variance is a squared
 * quantity, so in Q15 it quantises as the square of the signal level: with a
 * 1% amplitude ripple — an ordinary figure for CSI — the stored variance
 * lands on 1.64 and rounds to 2, a 22% error, and by 0.5% ripple it rounds
 * to zero and the feature disappears entirely. Taking the square root first
 * puts the quantity back on the same scale as the signal, where Q15 has
 * plenty of resolution.
 *
 * @param window   n_frames * n_sub amplitudes, frame-major.
 * @param out      n_sub * QCSI_STATS_PER_SUBCARRIER values, subcarrier-major:
 *                 out[k * 3 + QCSI_STAT_MEAN] and so on.
 * @return QDSP_OK, or QDSP_ERR_ARG on a null pointer or a zero dimension.
 */
qdsp_status_t qcsi_amplitude_stats(const q15_t *window, size_t n_frames,
                                   size_t n_sub, q15_t *out);

/**
 * Remove the per-subcarrier mean from a window, in place.
 *
 * Each subcarrier has its own static gain, set by antenna patterns and by
 * whatever is standing still in the room. That offset is large, carries no
 * motion information, and would dominate the DC bin of any Doppler
 * transform. Removing it before the FFT is what makes the Doppler spectrum
 * readable at all.
 */
qdsp_status_t qcsi_remove_static_component(q15_t *window, size_t n_frames,
                                           size_t n_sub);

/* ------------------------------------------------------------------ */
/* Doppler spectrum                                                    */
/* ------------------------------------------------------------------ */

/**
 * Power spectrum along time for one subcarrier.
 *
 * The caller supplies scratch space of n_fft complex samples; the routine
 * copies the subcarrier's time series into it, zero-pads if the window is
 * shorter than n_fft, and runs qdsp's Q15 FFT.
 *
 * Only the first half of the spectrum is written, since the input is real
 * and the spectrum is therefore conjugate symmetric.
 *
 * Note on scaling: qdsp's Q15 FFT scales its output by 1/n_fft, so the
 * power values are scaled by 1/n_fft^2. Comparisons across bins are
 * unaffected; comparisons against a float reference are not, and the tests
 * account for it explicitly.
 *
 * @param window   n_frames * n_sub amplitudes, frame-major.
 * @param sub      which subcarrier to transform.
 * @param n_fft    power of two, at least n_frames, at most QDSP_FFT_MAX_N.
 * @param scratch  n_fft complex samples, owned by the caller.
 * @param out      n_fft/2 power values.
 */
qdsp_status_t qcsi_doppler_power(const q15_t *window, size_t n_frames,
                                 size_t n_sub, size_t sub, uint16_t n_fft,
                                 qdsp_cplx_q15 *scratch, q31_t *out);

/**
 * Index of the strongest non-DC Doppler bin.
 *
 * A crude but useful summary of a window: roughly, how fast the dominant
 * motion is. Returns 0 if the spectrum is empty or flat.
 */
size_t qcsi_dominant_doppler_bin(const q31_t *power, size_t n_bins);

#ifdef __cplusplus
}
#endif

#endif /* QCSI_FEATURES_H */
