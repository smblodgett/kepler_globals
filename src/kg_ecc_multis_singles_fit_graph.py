"""
kg_ecc_multis_singles_fit_graph.py

The "proof" figure: your real best-fit eccentricity model matches the
multis' shape well but not the singles', because multis are tighter
(narrower true spread) and therefore far more decisive per planet in the
log-mean-exp sum than their headcount alone would suggest.

Left panel: multis marginal -- a weighted KDE of the real multis
eccentricity data (each real planet contributes total weight 1, split
across its own posterior draws, same convention used everywhere else in
this project) overlaid with the ACTUAL best-fit model PDF read straight from
best_fit.json. Expect these to sit almost on top of each other.

Right panel: singles residual -- weighted KDE of the real singles data MINUS
that same best-fit model PDF, as a function of e. A clear, systematic
residual (not noise scattered around zero) is the direct visual evidence
that the shared model is not describing singles well, even though it's
doing a good job for multis with the exact same curve.

No histogram bins anywhere on either side -- the empirical side is a KDE
(continuous, no bin-width choices) and the model side is the exact analytic
PDF, avoiding the unequal-bin-width artifact that misled the earlier binned
marginal plot.

Usage:
    python kg_ecc_multis_singles_fit_graph.py [path/to/final_kdc.csv] [model_run_folder]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from kg_ecc_population_split_quicktest import _load_ecc_with_population
from kg_probability_distributions import eccentricity_log_pdf
from kg_best_fit_loader import load_best_fit_ecc_theta
from kg_ecc_cdf_diagnostic import _model_cdf_grid
from kg_param_boundary_arrays import eccentricity_grid_array
from kg_plots import _format_edge


def _weighted_kde(e_values, weights, e_grid, bw_method=None):
    kde = gaussian_kde(e_values, weights=weights, bw_method=bw_method)
    return kde(e_grid)


def _model_pdf(theta, e_grid):
    alpha, lam, sigma_e = theta
    return np.exp(eccentricity_log_pdf(e_grid, alpha, lam, sigma_e))


def plot_multis_singles_fit(csv_path, model_run_folder=None,
                             save_path="ecc_multis_singles_fit.pdf",
                             e_max=0.99, n_grid=2000):
    df = _load_ecc_with_population(csv_path)
    df["planet_weight"] = 1.0 / df.groupby("unique_planet")["e"].transform("size")

    df_multi = df[~df["is_single"]]
    df_single = df[df["is_single"]]

    theta = load_best_fit_ecc_theta(model_run_folder)
    alpha, lam, sigma_e = theta
    print(f"Using best-fit (alpha, lam, sigma_e) = ({alpha:.4f}, {lam:.4f}, {sigma_e:.4f})")

    e_grid = np.linspace(1e-4, e_max, n_grid)
    model_pdf = _model_pdf(theta, e_grid)

    kde_multi = _weighted_kde(df_multi["e"].to_numpy(), df_multi["planet_weight"].to_numpy(), e_grid)
    kde_single = _weighted_kde(df_single["e"].to_numpy(), df_single["planet_weight"].to_numpy(), e_grid)

    fig, (ax_multi, ax_single) = plt.subplots(1, 2, figsize=(12, 5), dpi=300, facecolor="w")

    ax_multi.plot(e_grid, kde_multi, color="tab:green", linewidth=2.2, label="multis (weighted KDE)")
    ax_multi.plot(e_grid, model_pdf, color="k", linestyle="--", linewidth=2, label="best-fit model")
    ax_multi.set_title("Multis: shared model matches well", fontsize=11)
    ax_multi.set_xlabel("eccentricity")
    ax_multi.set_ylabel("density")
    ax_multi.legend(fontsize=9)

    residual = kde_single - model_pdf
    ax_single.axhline(0, color="gray", linewidth=1)
    ax_single.plot(e_grid, residual, color="tab:red", linewidth=2.2)
    ax_single.fill_between(e_grid, 0, residual, color="tab:red", alpha=0.25)
    ax_single.set_title("Singles: same model, systematic residual", fontsize=11)
    ax_single.set_xlabel("eccentricity")
    ax_single.set_ylabel("density residual (data $-$ model)")

    fig.suptitle(f"Best-fit eccentricity model (α={alpha:.3f}, λ={lam:.3f}, "
                 f"σ={sigma_e:.4f}) fits multis, not singles", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def plot_ecc_fit_marginals_and_residuals(csv_path, model_run_folder=None,
                                          save_path="ecc_fit_marginals_residuals.pdf",
                                          e_max=0.99, n_grid=2000):
    """
    Full symmetric version: marginal AND residual for BOTH populations,
    two rows (multis, singles) x two columns (marginal, residual), sharing
    y-axis scale within each column so the comparison is honest -- e.g. the
    two residual panels use the same y-limits, so "singles' residual is
    visibly bigger than multis'" is a real, unscaled visual comparison, not
    an artifact of matplotlib auto-scaling each panel independently.
    """
    df = _load_ecc_with_population(csv_path)
    df["planet_weight"] = 1.0 / df.groupby("unique_planet")["e"].transform("size")

    df_multi = df[~df["is_single"]]
    df_single = df[df["is_single"]]

    theta = load_best_fit_ecc_theta(model_run_folder)
    alpha, lam, sigma_e = theta
    print(f"Using best-fit (alpha, lam, sigma_e) = ({alpha:.4f}, {lam:.4f}, {sigma_e:.4f})")

    e_grid = np.linspace(1e-4, e_max, n_grid)
    model_pdf = _model_pdf(theta, e_grid)

    populations = [
        ("multis", df_multi, "tab:green"),
        ("singles", df_single, "tab:red"),
        ("combined", df, "tab:blue"),
    ]

    kdes, residuals = {}, {}
    for label, sub, _ in populations:
        e_vals = sub["e"].to_numpy()
        w = sub["planet_weight"].to_numpy()
        kdes[label] = _weighted_kde(e_vals, w, e_grid)
        residuals[label] = kdes[label] - model_pdf

    marg_ymax = max(kdes[l].max() for l, _, _ in populations)
    marg_ymax = max(marg_ymax, model_pdf.max()) * 1.05
    res_ymax = max(np.abs(residuals[l]).max() for l, _, _ in populations) * 1.1

    fig, axes = plt.subplots(len(populations), 2, figsize=(12, 4.5 * len(populations)),
                              dpi=300, facecolor="w", sharex=True)

    for row, (label, sub, color) in enumerate(populations):
        ax_marg = axes[row, 0]
        ax_marg.plot(e_grid, kdes[label], color=color, linewidth=2.2, label=f"{label} (weighted KDE)")
        ax_marg.plot(e_grid, model_pdf, color="k", linestyle="--", linewidth=2, label="best-fit model")
        ax_marg.set_ylim(0, marg_ymax)
        ax_marg.set_title(f"{label}: marginal", fontsize=11)
        ax_marg.set_ylabel("density")
        ax_marg.legend(fontsize=9)

        ax_res = axes[row, 1]
        ax_res.axhline(0, color="gray", linewidth=1)
        ax_res.plot(e_grid, residuals[label], color=color, linewidth=2.2)
        ax_res.fill_between(e_grid, 0, residuals[label], color=color, alpha=0.25)
        ax_res.set_ylim(-res_ymax, res_ymax)
        ax_res.set_title(f"{label}: residual (data $-$ model)", fontsize=11)
        ax_res.set_ylabel("density residual")

    for ax in axes[-1, :]:
        ax.set_xlabel("eccentricity")

    fig.suptitle(f"Best-fit eccentricity model (α={alpha:.3f}, λ={lam:.3f}, σ={sigma_e:.4f}): "
                 "marginals & residuals, multis vs singles", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def _binned_data_density(e_values, weights, edges):
    """Weighted histogram, normalized to a proper density: (weight fraction
    in bin) / (bin width). Uses the SAME edges as the rest of the project
    (eccentricity_grid_array), unlike a KDE -- no bandwidth choice, no
    boundary leakage, exactly the binning convention already used elsewhere."""
    counts, _ = np.histogram(e_values, bins=edges, weights=weights)
    widths = np.diff(edges)
    return counts / weights.sum() / widths


def _binned_model_density(theta, edges):
    """Model's average density per bin: exact CDF(hi) - CDF(lo) (via the same
    fine-grid numerical integration used everywhere else), divided by bin
    width -- NOT just the PDF evaluated at the bin center, which would be
    inaccurate for the wide bins here (some are 0.2 wide)."""
    e_fine, cdf_fine = _model_cdf_grid(theta)
    cdf_at_edges = np.interp(edges, e_fine, cdf_fine)
    widths = np.diff(edges)
    return np.diff(cdf_at_edges) / widths


def plot_ecc_fit_marginals_and_residuals_binned(csv_path, model_run_folder=None,
                                                 save_path="ecc_fit_marginals_residuals_binned.pdf"):
    """
    Same three rows (multis, singles, combined) x two columns (marginal,
    residual) as plot_ecc_fit_marginals_and_residuals, but using the
    project's actual eccentricity_grid_array bins instead of a KDE --
    plotted with each bar's TRUE width (not equal-spaced index positions),
    so bin-width differences are visible rather than hidden, unlike the
    earlier binned marginal plot that misled the multis-peak reading.
    """
    edges = np.array(eccentricity_grid_array)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    df = _load_ecc_with_population(csv_path)
    df["planet_weight"] = 1.0 / df.groupby("unique_planet")["e"].transform("size")
    df_multi = df[~df["is_single"]]
    df_single = df[df["is_single"]]

    theta = load_best_fit_ecc_theta(model_run_folder)
    alpha, lam, sigma_e = theta
    print(f"Using best-fit (alpha, lam, sigma_e) = ({alpha:.4f}, {lam:.4f}, {sigma_e:.4f})")

    model_density = _binned_model_density(theta, edges)

    populations = [
        ("multis", df_multi, "tab:green"),
        ("singles", df_single, "tab:red"),
        ("combined", df, "tab:blue"),
    ]

    data_density, residuals = {}, {}
    for label, sub, _ in populations:
        data_density[label] = _binned_data_density(sub["e"].to_numpy(), sub["planet_weight"].to_numpy(), edges)
        residuals[label] = data_density[label] - model_density

    marg_ymax = max(max(data_density[l].max() for l, _, _ in populations), model_density.max()) * 1.1
    res_ymax = max(np.abs(residuals[l]).max() for l, _, _ in populations) * 1.1

    fig, axes = plt.subplots(len(populations), 2, figsize=(12, 4.5 * len(populations)),
                              dpi=300, facecolor="w")

    for row, (label, sub, color) in enumerate(populations):
        ax_marg = axes[row, 0]
        ax_marg.bar(centers, data_density[label], width=widths, alpha=0.5, color=color,
                    edgecolor=color, label=f"{label} (binned data)")
        ax_marg.bar(centers, model_density, width=widths, fill=False, edgecolor="k",
                    linewidth=1.5, label="best-fit model (bin-avg)")
        ax_marg.set_ylim(0, marg_ymax)
        ax_marg.set_title(f"{label}: marginal", fontsize=11)
        ax_marg.set_ylabel("density")
        ax_marg.set_xticks(edges)
        ax_marg.set_xticklabels([_format_edge(e) for e in edges], rotation=45, fontsize=7)
        ax_marg.legend(fontsize=8)

        ax_res = axes[row, 1]
        ax_res.axhline(0, color="gray", linewidth=1)
        ax_res.bar(centers, residuals[label], width=widths, color=color, alpha=0.6,
                   edgecolor=color)
        ax_res.set_ylim(-res_ymax, res_ymax)
        ax_res.set_title(f"{label}: residual (data $-$ model)", fontsize=11)
        ax_res.set_ylabel("density residual")
        ax_res.set_xticks(edges)
        ax_res.set_xticklabels([_format_edge(e) for e in edges], rotation=45, fontsize=7)

    for ax in axes[-1, :]:
        ax.set_xlabel("eccentricity")

    fig.suptitle(f"Best-fit eccentricity model (α={alpha:.3f}, λ={lam:.3f}, σ={sigma_e:.4f}): "
                 "BINNED marginals & residuals (true bin widths), multis vs singles vs combined",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../data/final_kdc.csv"
    model_run_folder = sys.argv[2] if len(sys.argv) > 2 else None
    plot_ecc_fit_marginals_and_residuals_binned(csv_path, model_run_folder)
