"""
kg_best_fit_loader.py

Small shared utility so the eccentricity diagnostic scripts can pull the
ACTUAL fitted params from your real MCMC run's best_fit.json, instead of
each one silently re-optimizing its own standalone eccentricity-only proxy
fit. Mirrors the exact plotprops.txt-based loading convention kg_plots.py
and kg_ecc_population_diagnostics.py already use, so paths/keys stay
consistent across every script in this project.
"""

import os
import sys
import json

from kg_utilities import ReadJson


def _find_plotprops_filename():
    cwd = os.getcwd()
    if "src" in cwd:
        return "../runs/plotprops.txt"
    elif "runs" in cwd:
        return "plotprops.txt"
    elif "results" in cwd:
        return "plotprops.txt"
    else:
        sys.exit("Run this from a src, runs, or results directory (same rule as kg_plots.py).")


def load_best_fit_params(model_run_folder=None):
    """
    Returns the full 17-element best-fit params list from best_fit.json, read
    via the same plotprops.txt paths kg_plots.py/kg_ecc_population_diagnostics.py
    use. Pass model_run_folder to override plotprops.txt's own
    "model_run_folder" value (same convention as kg_plots.py's CLI argument).
    """
    plotprops = ReadJson(_find_plotprops_filename()).outProps()

    model_id = plotprops.get("model_id")
    model_run_folder = model_run_folder or plotprops.get("model_run_folder")

    best_guess_filename = (plotprops["best_guess_filename"] + f"model_{model_id}/"
                            + model_run_folder + "/best_fit.json")
    with open(best_guess_filename, "r") as f:
        saved = json.load(f)

    print(f"[kg_best_fit_loader] loaded params from {best_guess_filename} "
          f"(log_prob={saved.get('log_prob')})")
    return saved["params"]


def load_best_fit_ecc_theta(model_run_folder=None):
    """Returns just (alpha, lam, sigma_e) = params[14:17] -- the eccentricity
    shape parameters from your actual fitted model, matching the unpacking
    convention in kg_probability_distributions.get_probability_distributions
    / joint_log_intrinsic_density."""
    params = load_best_fit_params(model_run_folder)
    return tuple(params[14:17])


if __name__ == "__main__":
    model_run_folder = sys.argv[1] if len(sys.argv) > 1 else None
    theta = load_best_fit_ecc_theta(model_run_folder)
    print(f"Actual best-fit (alpha, lam, sigma_e) = {theta}")
