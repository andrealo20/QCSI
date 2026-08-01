#include "qcsi/classify.h"

qdsp_status_t qcsi_linear_init(qcsi_linear_model *m, const q15_t *weights,
                               const q63_t *bias, uint16_t n_classes,
                               uint16_t n_features)
{
    if (m == NULL || weights == NULL || n_classes == 0u || n_features == 0u) {
        return QDSP_ERR_ARG;
    }
    m->weights = weights;
    m->bias = bias;          /* NULL is allowed and means zero */
    m->n_classes = n_classes;
    m->n_features = n_features;
    return QDSP_OK;
}

qdsp_status_t qcsi_linear_scores(const qcsi_linear_model *m,
                                 const q15_t *features, q63_t *scores)
{
    uint16_t c, f;

    if (m == NULL || features == NULL || scores == NULL) {
        return QDSP_ERR_ARG;
    }

    for (c = 0u; c < m->n_classes; ++c) {
        const q15_t *w = &m->weights[(size_t)c * m->n_features];
        q63_t acc = (m->bias != NULL) ? m->bias[c] : 0;

        /* No saturation inside the loop: a class score must not depend on
           the order its features happen to be summed in. The accumulator is
           64 bits precisely so that it does not have to. */
        for (f = 0u; f < m->n_features; ++f) {
            acc = qdsp_mac_q15(acc, w[f], features[f]);
        }
        scores[c] = acc;
    }
    return QDSP_OK;
}

/** argmax with a defined tie-break, shared by predict and margin. */
static int argmax_scores(const qcsi_linear_model *m, const q15_t *features,
                         q63_t *scratch, q63_t *best_out, q63_t *second_out)
{
    uint16_t c, f;
    int best = -1;
    q63_t best_score = 0, second_score = 0;

    for (c = 0u; c < m->n_classes; ++c) {
        const q15_t *w = &m->weights[(size_t)c * m->n_features];
        q63_t acc = (m->bias != NULL) ? m->bias[c] : 0;

        for (f = 0u; f < m->n_features; ++f) {
            acc = qdsp_mac_q15(acc, w[f], features[f]);
        }
        if (scratch != NULL) {
            scratch[c] = acc;
        }

        if (best < 0 || acc > best_score) {
            /* Strictly greater, so an equal score never displaces an
               earlier class: ties go to the lower index, always. */
            second_score = (best < 0) ? acc : best_score;
            best_score = acc;
            best = (int)c;
        } else if (c == 1u || acc > second_score) {
            second_score = acc;
        }
    }

    if (best_out != NULL) *best_out = best_score;
    if (second_out != NULL) *second_out = second_score;
    return best;
}

int qcsi_linear_predict(const qcsi_linear_model *m, const q15_t *features,
                        q63_t *scratch)
{
    if (m == NULL || features == NULL) {
        return (int)QDSP_ERR_ARG;
    }
    return argmax_scores(m, features, scratch, NULL, NULL);
}

q63_t qcsi_linear_margin(const qcsi_linear_model *m, const q15_t *features,
                         q63_t *scratch)
{
    q63_t best = 0, second = 0;

    if (m == NULL || features == NULL || m->n_classes < 2u) {
        return 0;
    }
    (void)argmax_scores(m, features, scratch, &best, &second);
    return best - second;
}
