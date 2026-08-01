/**
 * Unit tests for the quantised linear classifier.
 *
 * The interesting properties here are not "does the dot product work" but
 * the ones that decide whether a parity test against the Python reference
 * means anything: that scores are order-independent, that ties break the
 * same way every time, and that the margin identifies the samples a small
 * perturbation can flip.
 */
#include "unity.h"
#include "qcsi/classify.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NCLS  6
#define NFEAT 24

static q15_t weights[NCLS * NFEAT];
static q63_t bias[NCLS];
static q15_t features[NFEAT];
static q63_t scores[NCLS];
static qcsi_linear_model model;

void setUp(void)
{
    int c, f;
    srand(7);
    for (c = 0; c < NCLS; ++c) {
        bias[c] = 0;
        for (f = 0; f < NFEAT; ++f) {
            weights[c * NFEAT + f] = (q15_t)((rand() % 8000) - 4000);
        }
    }
    for (f = 0; f < NFEAT; ++f) {
        features[f] = (q15_t)((rand() % 20000) - 10000);
    }
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_linear_init(&model, weights, bias, NCLS, NFEAT));
}

void tearDown(void) {}

/* ------------------------------------------------------------------ */
/* Contract                                                            */
/* ------------------------------------------------------------------ */

static void test_init_rejects_bad_arguments(void)
{
    qcsi_linear_model m;
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_linear_init(NULL, weights, bias, NCLS, NFEAT));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_linear_init(&m, NULL, bias, NCLS, NFEAT));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_linear_init(&m, weights, bias, 0, NFEAT));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_linear_init(&m, weights, bias, NCLS, 0));
    /* A null bias is allowed and means zero. */
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_linear_init(&m, weights, NULL, NCLS, NFEAT));
}

static void test_scores_match_a_double_reference(void)
{
    int c, f;
    double worst = 0.0;

    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_linear_scores(&model, features, scores));

    for (c = 0; c < NCLS; ++c) {
        double ref = (double)bias[c];
        for (f = 0; f < NFEAT; ++f) {
            ref += (double)weights[c * NFEAT + f] * (double)features[f];
        }
        /* The accumulator is exact for these magnitudes, so the match must
           be bit-for-bit rather than approximate. */
        TEST_ASSERT_EQUAL_INT64((int64_t)ref, scores[c]);
        if (fabs((double)scores[c] - ref) > worst) {
            worst = fabs((double)scores[c] - ref);
        }
    }
    TEST_ASSERT_EQUAL_DOUBLE(0.0, worst);
}

/**
 * A class score must not depend on the order its features are summed in.
 * With a saturating accumulator it would, and the model would silently
 * disagree with any reference implementation that iterated differently.
 */
static void test_scores_are_order_independent(void)
{
    static q15_t shuffled_w[NCLS * NFEAT];
    static q15_t shuffled_f[NFEAT];
    static q63_t shuffled_scores[NCLS];
    static int perm[NFEAT];
    qcsi_linear_model m2;
    int c, f, i;

    for (f = 0; f < NFEAT; ++f) perm[f] = f;
    for (f = NFEAT - 1; f > 0; --f) {
        int j = rand() % (f + 1);
        i = perm[f]; perm[f] = perm[j]; perm[j] = i;
    }

    for (f = 0; f < NFEAT; ++f) {
        shuffled_f[f] = features[perm[f]];
        for (c = 0; c < NCLS; ++c) {
            shuffled_w[c * NFEAT + f] = weights[c * NFEAT + perm[f]];
        }
    }

    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_linear_init(&m2, shuffled_w, bias, NCLS, NFEAT));
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_linear_scores(&model, features, scores));
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_linear_scores(&m2, shuffled_f, shuffled_scores));

    for (c = 0; c < NCLS; ++c) {
        TEST_ASSERT_EQUAL_INT64(scores[c], shuffled_scores[c]);
    }
}

static void test_predict_agrees_with_scores(void)
{
    int pred, c, best = 0;
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_linear_scores(&model, features, scores));
    for (c = 1; c < NCLS; ++c) {
        if (scores[c] > scores[best]) best = c;
    }
    pred = qcsi_linear_predict(&model, features, NULL);
    TEST_ASSERT_EQUAL_INT(best, pred);
}

static void test_scratch_is_optional_and_consistent(void)
{
    static q63_t scratch[NCLS];
    int with_scratch, without;
    int c;

    without = qcsi_linear_predict(&model, features, NULL);
    with_scratch = qcsi_linear_predict(&model, features, scratch);
    TEST_ASSERT_EQUAL_INT(without, with_scratch);

    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_linear_scores(&model, features, scores));
    for (c = 0; c < NCLS; ++c) {
        TEST_ASSERT_EQUAL_INT64(scores[c], scratch[c]);
    }
}

/* ------------------------------------------------------------------ */
/* Ties                                                                */
/* ------------------------------------------------------------------ */

/**
 * Ties must break towards the lower index, always. In floating point a tie
 * essentially never happens; once scores are quantised it does, and an
 * unspecified tie-break would make the C and Python implementations
 * disagree at random on precisely the samples a parity test examines.
 */
static void test_ties_break_towards_the_lower_index(void)
{
    static q15_t w[3 * 4] = {
        1000, 2000, 3000, 4000,     /* class 0 */
        1000, 2000, 3000, 4000,     /* class 1, identical */
        -100, -100, -100, -100      /* class 2, clearly worse */
    };
    static q15_t f[4] = { 5000, 5000, 5000, 5000 };
    static q63_t sc[3];
    qcsi_linear_model m;

    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_linear_init(&m, w, NULL, 3, 4));
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_linear_scores(&m, f, sc));
    TEST_ASSERT_EQUAL_INT64(sc[0], sc[1]);
    TEST_ASSERT_EQUAL_INT(0, qcsi_linear_predict(&m, f, NULL));
    /* And a tied top pair means zero margin. */
    TEST_ASSERT_EQUAL_INT64(0, qcsi_linear_margin(&m, f, NULL));
}

/* ------------------------------------------------------------------ */
/* Margin                                                              */
/* ------------------------------------------------------------------ */

static void test_margin_matches_the_top_two_scores(void)
{
    int c, best = 0, second = -1;
    q63_t margin;

    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_linear_scores(&model, features, scores));
    for (c = 1; c < NCLS; ++c) {
        if (scores[c] > scores[best]) best = c;
    }
    for (c = 0; c < NCLS; ++c) {
        if (c == best) continue;
        if (second < 0 || scores[c] > scores[second]) second = c;
    }

    margin = qcsi_linear_margin(&model, features, NULL);
    TEST_ASSERT_EQUAL_INT64(scores[best] - scores[second], margin);
    TEST_ASSERT_TRUE(margin >= 0);
}

/**
 * The margin has to actually predict fragility, otherwise reporting it
 * alongside the accuracy would be decoration. A perturbation of a given
 * size should flip low-margin samples and leave high-margin ones alone.
 */
static void test_small_margins_are_the_fragile_ones(void)
{
    static q15_t perturbed[NFEAT];
    int trial, flipped_low = 0, flipped_high = 0, low = 0, high = 0;

    /* Both constants below are measured rather than guessed. On this model
       the margin ranges from about 5.5e5 to 2.5e8 with a mean near 4.7e7,
       so the median-ish threshold is 4e7. A perturbation of +/-800 LSB per
       feature flips roughly 5% of samples, which is enough to compare the
       two groups without being so large that everything flips. An earlier
       version used +/-10 and flipped nothing at all in either group: the
       test passed no information, and asserting on it would have been
       asserting on noise. */
    const q63_t margin_threshold = 40000000;
    const int perturbation = 800;

    for (trial = 0; trial < 600; ++trial) {
        int f, before, after;
        q63_t margin;

        for (f = 0; f < NFEAT; ++f) {
            features[f] = (q15_t)((rand() % 20000) - 10000);
        }
        before = qcsi_linear_predict(&model, features, NULL);
        margin = qcsi_linear_margin(&model, features, NULL);

        for (f = 0; f < NFEAT; ++f) {
            perturbed[f] = qdsp_add_q15(
                features[f],
                (q15_t)((rand() % (2 * perturbation + 1)) - perturbation));
        }
        after = qcsi_linear_predict(&model, perturbed, NULL);

        if (margin < margin_threshold) {
            low++;
            if (after != before) flipped_low++;
        } else {
            high++;
            if (after != before) flipped_high++;
        }
    }

    printf("  [measured] flipped: %d/%d low-margin (%.1f%%), "
           "%d/%d high-margin (%.1f%%)\n",
           flipped_low, low, 100.0 * flipped_low / (low ? low : 1),
           flipped_high, high, 100.0 * flipped_high / (high ? high : 1));
    TEST_ASSERT_TRUE_MESSAGE(low > 0 && high > 0,
                             "test did not exercise both margin regimes");
    TEST_ASSERT_TRUE_MESSAGE(flipped_low > 0,
                             "no sample flipped: the perturbation is too "
                             "small for this test to say anything");
    /* Fragility must concentrate in the low-margin group. */
    TEST_ASSERT_TRUE_MESSAGE(
        (double)flipped_low / low > (double)flipped_high / high,
        "margin does not predict which samples flip");
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_init_rejects_bad_arguments);
    RUN_TEST(test_scores_match_a_double_reference);
    RUN_TEST(test_scores_are_order_independent);
    RUN_TEST(test_predict_agrees_with_scores);
    RUN_TEST(test_scratch_is_optional_and_consistent);
    RUN_TEST(test_ties_break_towards_the_lower_index);
    RUN_TEST(test_margin_matches_the_top_two_scores);
    RUN_TEST(test_small_margins_are_the_fragile_ones);
    return UNITY_END();
}
