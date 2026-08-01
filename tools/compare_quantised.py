#!/usr/bin/env python3
"""
Measure what fixed point costs, by comparing float and Q15 on the same
samples.

The comparison is paired, not two independent accuracies. Running both
pipelines over identical inputs and counting the samples whose prediction
*changes* is far more sensitive, because every sample that agrees
contributes no variance at all. On 2250 test samples with a few percent of
outcomes changing, this resolves differences of about half a percentage
point; comparing two accuracy figures resolves nothing below about two
points. That gap is why this script exists.

What is reported:

  agreement        fraction of samples where float and Q15 predict the same
  net accuracy     the accuracy of each pipeline, and the difference
  McNemar          whether the change in accuracy is more than chance
  by margin        where the disagreements sit in the margin distribution

The last one is the interesting one. Accuracy says how many samples were
lost; the margin distribution says which, and it should show that
disagreements concentrate where the top two class scores were close - the
samples a small perturbation was always going to flip. If they do not, the
loss is not quantisation noise and something else is wrong.

Usage:
    python3 tools/compare_quantised.py dataset_lab_150.mat --ant-a 0 --ant-b 1

Requires: numpy, scipy, scikit-learn.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcsi_data import extract_features, load_subjects   # noqa: E402
from build_baseline import apply_split, build_splits    # noqa: E402


def quantise(x, bits=15):
    """Symmetric quantisation to Q(bits), returning values and the scale."""
    scale = float(np.max(np.abs(x)))
    if scale == 0.0:
        return np.zeros_like(x, dtype=np.int64), 1.0
    limit = float(2 ** bits - 1)
    q = np.clip(np.round(x / scale * limit), -limit - 1, limit)
    return q.astype(np.int64), scale


def fixed_point_scores(x_q, w_q, b_q):
    """Exactly what qcsi_linear_scores() computes, in integers.

    int64 throughout, no saturation anywhere, so this mirrors the C
    accumulator rather than approximating it. If the two ever disagree, the
    cause is in the C code and not in this reference.
    """
    return x_q @ w_q.T + b_q[None, :]


def margins(scores):
    """Gap between the best and second-best score, per sample."""
    part = np.partition(scores, -2, axis=1)
    return part[:, -1] - part[:, -2]


def mcnemar(a_correct, b_correct):
    """Exact-ish McNemar on the discordant pairs.

    Only samples where exactly one pipeline is right carry information about
    which is better; the rest cancel. Returns (n01, n10, p) where n01 is the
    count that float got right and fixed point got wrong.
    """
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    # Normal approximation to the binomial with p=0.5, continuity-corrected.
    from math import erfc, sqrt
    z = (abs(n01 - n10) - 1.0) / sqrt(n)
    p = erfc(z / sqrt(2.0))
    return n01, n10, float(min(1.0, p))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--ant-a", type=int, default=0)
    ap.add_argument("--ant-b", type=int, default=1)
    ap.add_argument("--max-per-subject", type=int, default=None)
    ap.add_argument("--split-file", default="split.json")
    ap.add_argument("--point", default="within_subject",
                    choices=["within_subject", "cross_subject"],
                    help="operating point to measure on")
    ap.add_argument("--bits", type=int, default=15,
                    help="fixed-point word length to test")
    ap.add_argument("--sweep", action="store_true",
                    help="test several word lengths and report where the "
                         "pipeline starts to break")
    args = ap.parse_args()

    if not os.path.exists(args.split_file):
        raise SystemExit(f"{args.split_file} not found - run build_baseline.py "
                         "first so both tools use the same frozen split")
    with open(args.split_file) as f:
        splits = json.load(f)
    if args.point not in splits:
        raise SystemExit(f"'{args.point}' is not in {args.split_file}")

    subjects = load_subjects(args.path, max_per_subject=args.max_per_subject)
    print("\ncomputing features ...")
    feats = [extract_features(csi, args.ant_a, args.ant_b) for csi, _ in subjects]
    labels = [np.asarray(l).ravel() for _, l in subjects]
    xtr, ytr, xte, yte = apply_split(feats, labels, splits[args.point])
    yte = np.asarray(yte).ravel()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # The scaler is part of the model, so it is fitted on train and then
    # frozen: applying it to the test set is inference, not fitting.
    scaler = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(xtr),
                                                np.asarray(ytr).ravel())

    xte_s = scaler.transform(xte)
    float_scores = xte_s @ clf.coef_.T + clf.intercept_[None, :]
    float_pred = clf.classes_[np.argmax(float_scores, axis=1)]

    def run_at(bits):
        x_q, x_scale = quantise(xte_s, bits)
        w_q, w_scale = quantise(clf.coef_, bits)
        # The bias lands on the same scale as the accumulated dot product,
        # which is the product of the two quantisation scales.
        limit = float(2 ** bits - 1)
        b_q = np.round(clf.intercept_ / (x_scale * w_scale) * (limit ** 2))
        return clf.classes_[np.argmax(
            fixed_point_scores(x_q, w_q, b_q.astype(np.int64)), axis=1)]

    m_all = margins(float_scores)

    if args.sweep:
        print(f"\noperating point : {args.point}   n = {len(yte)}")
        print(f"float accuracy  : {float(np.mean(float_pred == yte)) * 100:.2f}%")
        print("\nword length sweep:")
        print("  bits   accuracy   change   disagree   low-margin  high-margin")
        med = float(np.median(m_all))
        for bits in (15, 12, 10, 8, 7, 6, 5, 4):
            pred = run_at(bits)
            acc = float(np.mean(pred == yte))
            ch = pred != float_pred
            lo = float(np.mean(ch[m_all < med])) * 100.0
            hi = float(np.mean(ch[m_all >= med])) * 100.0
            print(f"  Q{bits:<4} {acc * 100:7.2f}%  "
                  f"{(acc - float(np.mean(float_pred == yte))) * 100:+6.2f}   "
                  f"{int(ch.sum()):5d}      {lo:6.2f}%      {hi:6.2f}%")
        print("\nDisagreements should concentrate in the low-margin column. "
              "If they appear at high margins too, that is not precision "
              "loss - suspect a scaling or alignment error.")
        return 0

    fixed_pred = run_at(args.bits)

    # --- results --------------------------------------------------------
    n = len(yte)
    agree = float(np.mean(float_pred == fixed_pred))
    fl_ok = float_pred == yte
    fx_ok = fixed_pred == yte
    acc_f, acc_x = float(np.mean(fl_ok)), float(np.mean(fx_ok))

    print(f"\noperating point : {args.point}   n = {n}   word length: "
          f"Q{args.bits}")
    print(f"float accuracy  : {acc_f * 100:.2f}%")
    print(f"Q15 accuracy    : {acc_x * 100:.2f}%")
    print(f"difference      : {(acc_x - acc_f) * 100:+.2f} points")
    print(f"agreement       : {agree * 100:.2f}% of predictions identical")

    n_disagree = int(np.sum(float_pred != fixed_pred))
    if n_disagree == 0:
        print("\nThe two pipelines are identical on every test sample. At this "
              "feature dimension the quantisation noise never exceeds the "
              "decision margin.")
    else:
        se = float(np.sqrt((n_disagree / n) * (1 - n_disagree / n) / n))
        print(f"                  ({n_disagree} disagreements, "
              f"standard error {se * 100:.2f} points)")

    n01, n10, p = mcnemar(fl_ok, fx_ok)
    print(f"\nMcNemar         : float right / Q15 wrong: {n01}, "
          f"the other way: {n10}, p = {p:.3f}")
    if p > 0.05:
        print("                  the accuracy change is not distinguishable "
              "from chance")

    # --- where the disagreements sit ------------------------------------
    m = m_all
    if n_disagree > 0:
        changed = float_pred != fixed_pred
        edges = np.quantile(m, [0, 0.25, 0.5, 0.75, 1.0])
        print("\nby decision margin (float), quartiles:")
        for i in range(4):
            sel = (m >= edges[i]) & (m <= edges[i + 1] if i == 3
                                     else m < edges[i + 1])
            if sel.sum() == 0:
                continue
            rate = float(np.mean(changed[sel]))
            print(f"  Q{i + 1}: margin {edges[i]:8.2f}-{edges[i + 1]:8.2f}  "
                  f"{int(changed[sel].sum()):4d}/{int(sel.sum()):4d} changed "
                  f"({rate * 100:5.2f}%)")
        lo = float(np.mean(changed[m < np.median(m)]))
        hi = float(np.mean(changed[m >= np.median(m)]))
        print(f"\n  below-median margin: {lo * 100:.2f}% changed, "
              f"above: {hi * 100:.2f}%")
        if lo <= hi:
            print("  WARNING: disagreements are not concentrated at small "
                  "margins. That is not what quantisation noise looks like - "
                  "suspect a scaling or alignment error rather than precision "
                  "loss.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
