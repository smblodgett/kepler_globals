"""
kg_ecc_cdf_diagnostic.py

Binless replacement for the histogram/marginal-bar comparisons. The
point-process likelihood never bins anything -- every planet's posterior
draws are scored via the closed-form eccentricity_log_pdf at their own exact
value -- so a fair diagnostic should compare distributions the same way:
empirical CDF vs. model CDF, with no bin-edge choices (and therefore no
unequal-bin-width artifacts like the one that misled the earlier histogram
read).

For multis and singles (each fit its own (alpha, lam, sigma_e)) plus the
shared fit, this:
  1. Builds the model CDF by numerically integrating rayleigh_exponential
     over a fine eccentricity grid (no closed-form CDF exists for the
     mixture, but the integral is cheap and exact enough for comparison).
  2. Builds the weighted empirical CDF of the real data -- each real planet
     contributes total weight 1 spread across its own posterior draws
     (matching how the likelihood pools draws), not a naive unweighted ECDF
     of every CSV row.
  3. Plots empirical vs. model CDFs (full range + a zoomed low-e panel,
     since that's the region in question) and reports the weighted
     Kolmogorov-Smirnov statistic -- a single, bin-free number for how well
     each model matches its data.

Usage:
    python kg_ecc_cdf_diagnostic.py [path/to/final_kdc.csv]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from kg_probability_distributions import eccentricity_log_pdf
from kg_ecc_population_split_quicktest import _load_ecc_with_population, fit_ecc_shape


def _model_cdf_grid(theta, e_grid=None):
    if e_grid is None:
        e_grid = np.linspace(0.0, 0.99, 200_000)
    alpha, lam, sigma_e = theta
    pdf = np.exp(eccentricity_log_pdf(e_grid, alpha, lam, sigma_e))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(e_grid))])
    cdf /= cdf[-1]  # normalize away tiny numerical drift from the finite grid
    return e_grid, cdf


def _weighted_ecdf(e_values, weights):
    order = np.argsort(e_values)
    e_sorted = e_values[order]
    cum_w = np.cumsum(weights[order])
    cum_w /= cum_w[-1]
    return e_sorted, cum_w


def weighted_ks_statistic(e_values, weights, theta):
    e_grid, model_cdf = _model_cdf_grid(theta)
    e_sorted, ecdf = _weighted_ecdf(e_values, weights)
    model_cdf_at_data = np.interp(e_sorted, e_grid, model_cdf)
    return np.max(np.abs(ecdf - model_cdf_at_data))


def plot_ecc_cdf_comparison(csv_path, save_path="ecc_cdf_comparison.pdf", zoom_max=0.1):
    df = _load_ecc_with_population(csv_path)
    df["planet_weight"] = 1.0 / df.groupby("unique_planet")["e"].transform("size")

    df_multi = df[~df["is_single"]]
    df_single = df[df["is_single"]]

    theta_multi, _ = fit_ecc_shape(df_multi)
    theta_single, _ = fit_ecc_shape(df_single)
    theta_shared, _ = fit_ecc_shape(df)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    e_grid_shared, cdf_shared = _model_cdf_grid(theta_shared)
    for ax in (ax_full, ax_zoom):
        ax.plot(e_grid_shared, cdf_shared, color="k", linestyle=":", linewidth=1.5,
                label="shared-fit model")

    for label, sub, theta, color in [
        ("multis", df_multi, theta_multi, "tab:green"),
        ("singles", df_single, theta_single, "tab:red"),
    ]:
        e_vals = sub["e"].to_numpy()
        w = sub["planet_weight"].to_numpy()
        e_sorted, ecdf = _weighted_ecdf(e_vals, w)
        e_grid, model_cdf = _model_cdf_grid(theta)

        ks_own = weighted_ks_statistic(e_vals, w, theta)
        ks_shared = weighted_ks_statistic(e_vals, w, theta_shared)
        print(f"{label}: KS vs own-fit = {ks_own:.4f}, KS vs shared-fit = {ks_shared:.4f}")

        for ax in (ax_full, ax_zoom):
            ax.step(e_sorted, ecdf, where="post", color=color, linewidth=1.5,
                    label=f"{label} -- empirical (weighted)")
            ax.plot(e_grid, model_cdf, color=color, linestyle="--",
                    label=f"{label} -- own-fit model")

    ax_full.set_xlim(0, 1)
    ax_full.set_title("full range")
    ax_zoom.set_xlim(0, zoom_max)
    ax_zoom.set_title(f"zoomed: e in [0, {zoom_max}]")
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("eccentricity")
        ax.set_ylabel("CDF")
    ax_full.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../data/final_kdc.csv"
    plot_ecc_cdf_comparison(csv_path)
