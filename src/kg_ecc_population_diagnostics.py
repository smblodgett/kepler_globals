"""
kg_ecc_population_diagnostics.py

Two complementary ways to see how the singles vs. multis populations pull on
the (currently single, shared) eccentricity model.

1. plot_ecc_marginal_by_population(...) -- same "physical catalog" /
   "observed catalog" model bars as kg_plots.pointprocess_1D_marginal_plot,
   but with the "data" bars split into singles vs multis instead of combined.
   Shows visually whether the one shared eccentricity curve is a reasonable
   description of EITHER population, or a compromise between two different
   shapes.

2. ecc_logL_contribution_by_population(...) -- computes each real planet's
   per-planet term (log-mean-exp of the eccentricity factor alone over that
   planet's own posterior draws, exactly the same math
   kg_likelihood.parametric_log_likelihood_pointprocess uses for the full 5-D
   density, but restricted to just eccentricity_log_pdf so period/mass/radius
   don't dilute the picture), split by population. Evaluate this at two
   nearby parameter points (e.g. current best-fit sigma_e vs. a perturbed
   sigma_e) and compare how much each population's summed term moves --
   whichever population's contribution changes more is the one actually
   driving that parameter's gradient in the MCMC.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kg_plots import pointprocess_synthetic_catalog, _format_edge
from kg_param_boundary_arrays import eccentricity_grid_array
from kg_probability_distributions import eccentricity_log_pdf, profile_optimal_gamma0


def _load_catalog_with_population(csv_path):
    cols = ["Period_days", "M_pE", "R_pE", "e", "omega", "unique_planet"]
    df = pd.read_csv(csv_path, usecols=cols)
    df["is_single"] = df["unique_planet"].str.endswith("_0.1")
    # each real planet contributes total weight 1, spread across its draws --
    # matches kg_plots._observed_draw_weights / how the likelihood pools draws
    df["planet_weight"] = 1.0 / df.groupby("unique_planet")["e"].transform("size")
    return df


def plot_ecc_marginal_by_population(params, stellar_info, voxel_grid, csv_path,
                                     min_density=0.01, max_density=10.0,
                                     synthetic_multiplier=200,
                                     y_axis_scale="log", mode="save",
                                     save_path="pointprocess_1D_marginal_eccentricity_by_pop.pdf"):
    edges = np.array(eccentricity_grid_array)

    df = _load_catalog_with_population(csv_path)
    n_planets = df["unique_planet"].nunique()

    trimmed_catalog, completeness_weights, density_mask = pointprocess_synthetic_catalog(
        params, stellar_info, voxel_grid, min_density=min_density, max_density=max_density
    )
    # trimmed_catalog columns are (radius, period, mass, e, omega)
    synth_e = trimmed_catalog[density_mask, 3]
    synth_completeness_weights = completeness_weights[density_mask]
    synth_physical_weights = np.ones_like(synth_completeness_weights)

    Lambda_tilde = np.sum(synth_completeness_weights) / synthetic_multiplier
    Gamma0_opt = profile_optimal_gamma0(n_planets, Lambda_tilde)

    physical_count, _ = np.histogram(synth_e, bins=edges, weights=synth_physical_weights)
    observed_count, _ = np.histogram(synth_e, bins=edges, weights=synth_completeness_weights)
    physical_count = physical_count * (Gamma0_opt / synthetic_multiplier)
    observed_count = observed_count * (Gamma0_opt / synthetic_multiplier)

    multi_mask = ~df["is_single"]
    single_mask = df["is_single"]
    data_multi, _ = np.histogram(df.loc[multi_mask, "e"], bins=edges,
                                  weights=df.loc[multi_mask, "planet_weight"])
    data_single, _ = np.histogram(df.loc[single_mask, "e"], bins=edges,
                                   weights=df.loc[single_mask, "planet_weight"])

    x = np.arange(len(edges) - 1)
    plt.figure(dpi=300, facecolor="w")
    plt.bar(x, physical_count, alpha=0.35, label="physical catalog (model, intrinsic)")
    plt.bar(x, observed_count, alpha=0.35, label="observed catalog (model, detected)")
    plt.bar(x, data_multi, alpha=0.6, label="data -- multis")
    plt.bar(x, data_single, alpha=0.6, label="data -- singles", bottom=data_multi)
    plt.xticks(np.arange(len(edges)) - 0.5, [_format_edge(e) for e in edges], rotation=45)
    plt.yscale(y_axis_scale)
    plt.xlabel("eccentricity")
    plt.ylabel("planet count")
    plt.legend(fontsize=8)
    plt.title(f"Eccentricity marginal by population\n"
              rf"$\Gamma_0$={Gamma0_opt:.3g}, n_planets={n_planets}", fontsize=10)
    if mode == "save":
        plt.savefig(save_path)
    elif mode == "show":
        plt.show()
    plt.close()


def ecc_logL_contribution_by_population(params, csv_path):
    """
    Isolates the eccentricity-only piece of each real planet's per-planet
    term: term_e = log(mean_k[ f_e(e_jk) ]) over planet j's own posterior
    draws k, using params[14:17] = (alpha, lam, sigma_e). This is the exact
    log-mean-exp pooling kg_likelihood.py uses, just restricted to the
    eccentricity factor so period/mass/radius don't dilute the comparison.

    Returns a per-planet DataFrame (unique_planet, is_single, term) and
    prints a population-level summary (count, mean, sum). Run this at two
    nearby parameter points and diff the "sum" column by population to see
    which population's data is actually driving movement in
    (alpha, lam, sigma_e).
    """
    alpha, lam, sigma_e = params[14], params[15], params[16]

    df = _load_catalog_with_population(csv_path)
    df["log_f_e"] = eccentricity_log_pdf(df["e"].to_numpy(), alpha, lam, sigma_e)

    rows = []
    for planet_id, group in df.groupby("unique_planet", sort=False):
        vals = group["log_f_e"].to_numpy()
        m = vals.max()
        term = m + np.log(np.mean(np.exp(vals - m)))
        rows.append({
            "unique_planet": planet_id,
            "is_single": bool(group["is_single"].iloc[0]),
            "term": term,
        })
    result = pd.DataFrame(rows)

    summary = result.groupby("is_single")["term"].agg(["count", "mean", "sum"])
    print(summary)
    return result


if __name__ == "__main__":
    # Mirrors kg_plots.py's own loading convention (same plotprops.txt, same
    # voxel_grid/stellar_info/best_fit.json paths), so this runs the same way:
    #     cd src
    #     python kg_ecc_population_diagnostics.py [model_run_folder]
    # model_run_folder is optional -- defaults to plotprops.txt's "model_run_folder"
    # the same way kg_plots.py's model_run_folder_argv does.
    import os
    import sys
    import json

    from kg_utilities import ReadJson
    from kg_grid_object_hook import grid_object_hook

    cwd = os.getcwd()
    if "src" in cwd:
        plotprops_filename = "../runs/plotprops.txt"
    elif "runs" in cwd:
        plotprops_filename = "plotprops.txt"
    elif "results" in cwd:
        plotprops_filename = "plotprops.txt"
    else:
        sys.exit("Run this from a src, runs, or results directory (same rule as kg_plots.py).")

    plotprops = ReadJson(plotprops_filename).outProps()

    model_id = plotprops.get("model_id")
    model_run_folder = sys.argv[1] if len(sys.argv) > 1 else plotprops.get("model_run_folder")
    observed_catalog_filename = plotprops.get("observed_catalog_filename", "../data/final_kdc.csv")

    print(f"[kg_ecc_population_diagnostics] model_id={model_id}, model_run_folder={model_run_folder}")

    with open(plotprops["voxel_json_filename"], "r") as f:
        voxel_grid = json.load(f, object_hook=grid_object_hook)

    stellar_df = pd.read_csv(plotprops["processed_stellar_data_filename"])
    synthetic_multiplier = 200  # matches kg_plots.py's own pointprocess-marginal-plot convention
    stellar_info = stellar_df[["Rad", "Mass"]].to_numpy(dtype=np.float32)
    stellar_info = np.repeat(stellar_info, synthetic_multiplier, axis=0)

    best_guess_filename = (plotprops["best_guess_filename"] + f"model_{model_id}/"
                            + model_run_folder + "/best_fit.json")
    with open(best_guess_filename, "r") as f:
        params = json.load(f)["params"]
    print(f"Loaded best-fit params from {best_guess_filename}")

    plot_ecc_marginal_by_population(params, stellar_info, voxel_grid, observed_catalog_filename,
                                     synthetic_multiplier=synthetic_multiplier, mode="save")
    print("Saved pointprocess_1D_marginal_eccentricity_by_pop.pdf")

    ecc_logL_contribution_by_population(params, observed_catalog_filename)
