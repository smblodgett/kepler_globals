"""
kg_ecc_population_split_quicktest.py

Fast (non-MCMC) check for whether a multis/singles-split eccentricity model
is actually worth building, before threading a population covariate through
the full point-process likelihood and re-running MCMC.

This fits the SAME eccentricity_log_pdf(e; alpha, lam, sigma_e) machinery
kg_likelihood.py uses -- the per-planet log-mean-exp over each planet's own
posterior draws -- but as a standalone MLE over just the eccentricity data,
in two ways:

  1. one shared (alpha, lam, sigma_e) fit to every planet at once
  2. two separate (alpha, lam, sigma_e) triples, one fit to multis only and
     one fit to singles only

and compares summed log-likelihoods via a likelihood-ratio test (the split
model has 3 extra free parameters, so it will always fit at least as well --
the question is whether the improvement is big enough to matter) plus AIC.

Caveat: this ignores the completeness/Lambda_tilde coupling (eccentricity
shape also affects the synthetic-catalog-based rate normalization in the
real likelihood -- see kg_likelihood.parametric_log_likelihood_pointprocess),
so it's a proxy for "is there real signal in the data for splitting," not a
stand-in for re-running the actual MCMC. But it runs in seconds and should
give a clear go/no-go before doing the more invasive change.

Usage:
    python kg_ecc_population_split_quicktest.py [path/to/final_kdc.csv]
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from kg_probability_distributions import eccentricity_log_pdf


def _load_ecc_with_population(csv_path):
    df = pd.read_csv(csv_path, usecols=["e", "unique_planet"])
    df["is_single"] = df["unique_planet"].str.endswith("_0.1")
    return df


def _ecc_negLogL(theta, e_values, group_starts, group_counts):
    """Negative summed per-planet log-mean-exp eccentricity term, vectorized
    with reduceat exactly like kg_likelihood.py's grouped log-sum-exp."""
    alpha, lam, sigma_e = theta
    log_f = eccentricity_log_pdf(e_values, alpha, lam, sigma_e)

    seg_max = np.maximum.reduceat(log_f, group_starts)
    shifted = log_f - np.repeat(seg_max, group_counts)
    seg_sumexp = np.add.reduceat(np.exp(shifted), group_starts)
    term_per_planet = seg_max + np.log(seg_sumexp) - np.log(group_counts)

    return -np.sum(term_per_planet)


def _grouped_arrays(df):
    df = df.sort_values("unique_planet", kind="mergesort")
    e_values = df["e"].to_numpy(dtype=np.float64)
    group_ids = df["unique_planet"].to_numpy()
    _, group_starts, group_counts = np.unique(group_ids, return_index=True, return_counts=True)
    return e_values, group_starts.astype(np.int64), group_counts.astype(np.int64)


def fit_ecc_shape(df, x0=(0.5, 3.0, 0.05)):
    e_values, group_starts, group_counts = _grouped_arrays(df)
    bounds = [(1e-3, 1 - 1e-3), (1e-3, 50.0), (1e-3, 2.0)]
    result = minimize(_ecc_negLogL, x0=x0, args=(e_values, group_starts, group_counts),
                       method="L-BFGS-B", bounds=bounds)
    return result.x, -result.fun


def compare_shared_vs_split(csv_path):
    df = _load_ecc_with_population(csv_path)
    df_multi = df[~df["is_single"]]
    df_single = df[df["is_single"]]

    theta_shared, logL_shared = fit_ecc_shape(df)
    theta_multi, logL_multi = fit_ecc_shape(df_multi)
    theta_single, logL_single = fit_ecc_shape(df_single)
    logL_split = logL_multi + logL_single

    dof = 3  # split model has 3 extra free parameters
    lr_stat = 2 * (logL_split - logL_shared)
    p_value = chi2.sf(lr_stat, df=dof)

    aic_shared = 2 * 3 - 2 * logL_shared
    aic_split = 2 * 6 - 2 * logL_split

    print(f"n_multi={len(df_multi['unique_planet'].unique())}, "
          f"n_single={len(df_single['unique_planet'].unique())}")
    print(f"shared  (alpha, lam, sigma_e) = {np.round(theta_shared, 4)}   logL = {logL_shared:.2f}")
    print(f"multi   (alpha, lam, sigma_e) = {np.round(theta_multi, 4)}   logL = {logL_multi:.2f}")
    print(f"single  (alpha, lam, sigma_e) = {np.round(theta_single, 4)}   logL = {logL_single:.2f}")
    print(f"split total logL = {logL_split:.2f}  vs shared logL = {logL_shared:.2f}  "
          f"(delta = {logL_split - logL_shared:.2f})")
    print(f"likelihood-ratio stat = {lr_stat:.2f} on {dof} dof, p = {p_value:.3g}")
    print(f"AIC shared = {aic_shared:.2f}, AIC split = {aic_split:.2f} (lower is better, "
          f"delta AIC = {aic_shared - aic_split:.2f})")

    return {
        "theta_shared": theta_shared, "logL_shared": logL_shared,
        "theta_multi": theta_multi, "logL_multi": logL_multi,
        "theta_single": theta_single, "logL_single": logL_single,
        "lr_stat": lr_stat, "p_value": p_value,
        "aic_shared": aic_shared, "aic_split": aic_split,
    }


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../data/final_kdc.csv"
    compare_shared_vs_split(csv_path)
