"""
kg_ecc_prior_correction_test.py

Tests whether the multis' anomalous KS result (own-fit matching the pooled
empirical distribution WORSE than the shared fit does -- see
kg_ecc_cdf_diagnostic.py) is explained by an uncorrected reference-prior
mismatch: PhoDyMM's TTV fits sample (e*cos(w), e*sin(w)) rather than (e, w)
directly. The Jacobian of that transform is dx dy = e de dw, so a flat prior
in (ecosw, esinw) implies an IMPLIED prior on e of pi_ref(e) ~ e -- not flat.

Standard hierarchical population-inference identity (Hogg, Myers & Bovy
2010): the correct per-planet marginal contribution is the log-mean-exp of
f_pop(e_jk) / pi_ref(e_jk) over that planet's posterior draws, NOT the plain
log-mean-exp of f_pop(e_jk) that kg_likelihood.py currently computes. Skipping
that division lets the reference prior's own shape (~e, vanishing at e=0)
leak into what the population fit interprets as real signal.

This only applies to the multis (PhoDyMM-derived posteriors) -- singles were
already sampled with rng.uniform(0, 0.99), a genuinely flat prior on e, in
kg_initialize_voxel_grid.sample_eccentricity_omega, so they need no
correction.

Usage:
    python kg_ecc_prior_correction_test.py [path/to/final_kdc.csv]
"""

import sys
import numpy as np

from kg_ecc_population_split_quicktest import _load_ecc_with_population, _grouped_arrays, fit_ecc_shape
from kg_ecc_cdf_diagnostic import _weighted_ecdf, _model_cdf_grid, weighted_ks_statistic
from kg_probability_distributions import eccentricity_log_pdf
from scipy.optimize import minimize


def _ecc_negLogL_prior_corrected(theta, e_values, group_starts, group_counts):
    """Same grouped log-mean-exp as kg_likelihood.py, but with each draw's
    log-density corrected by -log(e) to divide out the implied pi_ref(e) ~ e
    reference prior from the ecos(w)/esin(w) sampling."""
    alpha, lam, sigma_e = theta
    log_f = eccentricity_log_pdf(e_values, alpha, lam, sigma_e) - np.log(np.maximum(e_values, 1e-12))

    seg_max = np.maximum.reduceat(log_f, group_starts)
    shifted = log_f - np.repeat(seg_max, group_counts)
    seg_sumexp = np.add.reduceat(np.exp(shifted), group_starts)
    term_per_planet = seg_max + np.log(seg_sumexp) - np.log(group_counts)

    return -np.sum(term_per_planet)


def fit_ecc_shape_prior_corrected(df, x0=(0.5, 3.0, 0.05)):
    e_values, group_starts, group_counts = _grouped_arrays(df)
    bounds = [(1e-3, 1 - 1e-3), (1e-3, 50.0), (1e-3, 2.0)]
    result = minimize(_ecc_negLogL_prior_corrected, x0=x0, args=(e_values, group_starts, group_counts),
                       method="L-BFGS-B", bounds=bounds)
    return result.x, -result.fun


def compare_corrected_vs_uncorrected(csv_path):
    df = _load_ecc_with_population(csv_path)
    df_multi = df[~df["is_single"]]

    theta_uncorrected, _ = fit_ecc_shape(df_multi)
    theta_corrected, _ = fit_ecc_shape_prior_corrected(df_multi)

    e_vals = df_multi["e"].to_numpy()
    weights = np.ones_like(e_vals)  # unweighted here -- one row per posterior draw, matching
    # the group structure already used for fitting (each planet's total pull
    # in the *fit* comes from its own log-mean-exp, but for the plain
    # empirical-CDF comparison every draw counts once, consistent with
    # kg_ecc_cdf_diagnostic's per-planet weighting)
    planet_weight = 1.0 / df_multi.groupby("unique_planet")["e"].transform("size")
    w = planet_weight.to_numpy()

    ks_uncorrected = weighted_ks_statistic(e_vals, w, theta_uncorrected)
    ks_corrected = weighted_ks_statistic(e_vals, w, theta_corrected)

    print(f"uncorrected multis fit: (alpha, lam, sigma_e) = {np.round(theta_uncorrected, 4)}, "
          f"KS vs empirical = {ks_uncorrected:.4f}")
    print(f"prior-corrected multis fit: (alpha, lam, sigma_e) = {np.round(theta_corrected, 4)}, "
          f"KS vs empirical = {ks_corrected:.4f}")
    print()
    if ks_corrected < ks_uncorrected:
        print(f"Correction IMPROVES the match ({ks_corrected:.4f} < {ks_uncorrected:.4f}) -- "
              "supports the ecos(w)/esin(w) reference-prior mismatch as a real contributor.")
    else:
        print(f"Correction does NOT improve the match ({ks_corrected:.4f} >= {ks_uncorrected:.4f}) -- "
              "the prior mismatch may not be the (or the only) driver; worth checking the "
              "per-planet log-mean-exp exploit angle instead/also.")

    return theta_uncorrected, theta_corrected, ks_uncorrected, ks_corrected


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../data/final_kdc.csv"
    compare_corrected_vs_uncorrected(csv_path)
