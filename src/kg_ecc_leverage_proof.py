"""
kg_ecc_leverage_proof.py

Two direct tests of the specific mechanism claimed for why the shared
eccentricity fit ends up "too peaked": that the tightly-constrained multis
dominate the sum disproportionately to their raw count, because a genuinely
narrow population lets f_pop get very tall (huge per-point log-density) while
a genuinely broad population (singles) mechanically caps how tall f_pop can
ever get anywhere -- so the shared fit should look like it's protecting the
multis' peak rather than splitting the difference in proportion to how many
planets each group has (887 multis vs 2485 singles).

1. cross_evaluate_costs() -- the direct "swap test": evaluate each
   population's own summed logL using the OTHER population's fitted
   (alpha, lam, sigma_e), and compare the per-planet cost of that swap.
   If multis lose far more per planet from being forced into singles' broad
   shape than singles lose from being forced into multis' narrow shape,
   that asymmetry -- not planet count -- is what's pulling the shared fit
   toward the tight peak.

2. equal_n_subsample_test() -- holds sample SIZE fixed and equal (887 vs
   887, subsampling singles down to match) and refits the shared model on
   that balanced set. If the balanced-count shared fit still sits close to
   the multis-only fit rather than splitting the difference with the
   singles-only fit, that rules out "there just happen to be more multis
   near the peak" as an alternative (count-driven) explanation.

Usage:
    python kg_ecc_leverage_proof.py [path/to/final_kdc.csv]
"""

import sys
import numpy as np
import pandas as pd

from kg_ecc_population_split_quicktest import (
    _load_ecc_with_population,
    _grouped_arrays,
    _ecc_negLogL,
    fit_ecc_shape,
)


def cross_evaluate_costs(csv_path):
    df = _load_ecc_with_population(csv_path)
    df_multi = df[~df["is_single"]]
    df_single = df[df["is_single"]]
    n_multi = df_multi["unique_planet"].nunique()
    n_single = df_single["unique_planet"].nunique()

    theta_multi, logL_multi_own = fit_ecc_shape(df_multi)
    theta_single, logL_single_own = fit_ecc_shape(df_single)

    e_m, gs_m, gc_m = _grouped_arrays(df_multi)
    e_s, gs_s, gc_s = _grouped_arrays(df_single)

    # each population's logL using the OTHER population's preferred shape
    logL_multi_at_single_theta = -_ecc_negLogL(theta_single, e_m, gs_m, gc_m)
    logL_single_at_multi_theta = -_ecc_negLogL(theta_multi, e_s, gs_s, gc_s)

    cost_to_multi = logL_multi_own - logL_multi_at_single_theta    # > 0: multis' loss using singles' shape
    cost_to_single = logL_single_own - logL_single_at_multi_theta  # > 0: singles' loss using multis' shape

    cost_to_multi_per = cost_to_multi / n_multi
    cost_to_single_per = cost_to_single / n_single

    print("=== cross_evaluate_costs (swap test) ===")
    print(f"theta_multi (own fit)  = {np.round(theta_multi, 4)}")
    print(f"theta_single (own fit) = {np.round(theta_single, 4)}")
    print()
    print(f"multis:  own-fit logL={logL_multi_own:.2f}  "
          f"at singles'-theta logL={logL_multi_at_single_theta:.2f}  "
          f"cost={cost_to_multi:.2f}  ({cost_to_multi_per:.4f} nats/planet)")
    print(f"singles: own-fit logL={logL_single_own:.2f}  "
          f"at multis'-theta logL={logL_single_at_multi_theta:.2f}  "
          f"cost={cost_to_single:.2f}  ({cost_to_single_per:.4f} nats/planet)")
    print()
    print(f"asymmetry ratio (per-planet cost to multis / per-planet cost to singles) "
          f"= {cost_to_multi_per / cost_to_single_per:.2f}")
    print("(a ratio well above 1 means: forcing multis to use singles' broad shape hurts "
          "each multi far more than forcing singles to use multis' narrow shape hurts each "
          "single -- exactly the asymmetric leverage that pulls the shared fit toward the "
          "tight peak, independent of the 887-vs-2485 count difference.)")

    return {
        "theta_multi": theta_multi, "theta_single": theta_single,
        "cost_to_multi": cost_to_multi, "cost_to_single": cost_to_single,
        "cost_to_multi_per_planet": cost_to_multi_per,
        "cost_to_single_per_planet": cost_to_single_per,
    }


def equal_n_subsample_test(csv_path, n_reps=50, seed=0):
    df = _load_ecc_with_population(csv_path)
    df_multi = df[~df["is_single"]]
    df_single = df[df["is_single"]]

    theta_multi, _ = fit_ecc_shape(df_multi)
    theta_single, _ = fit_ecc_shape(df_single)

    single_ids = df_single["unique_planet"].unique()
    n = df_multi["unique_planet"].nunique()  # 887, matched count

    rng = np.random.default_rng(seed)
    sigma_es, alphas, lams = [], [], []
    for _ in range(n_reps):
        chosen = rng.choice(single_ids, size=n, replace=False)
        balanced = pd.concat([df_multi, df_single[df_single["unique_planet"].isin(chosen)]])
        theta, _ = fit_ecc_shape(balanced)
        alphas.append(theta[0]); lams.append(theta[1]); sigma_es.append(theta[2])

    sigma_es = np.array(sigma_es)

    print("\n=== equal_n_subsample_test (count-matched balanced refit) ===")
    print(f"n = {n} multis + {n} randomly subsampled singles, {n_reps} repeats")
    print(f"balanced shared sigma_e: mean={sigma_es.mean():.4f}, std={sigma_es.std():.4f}")
    print(f"  multis-only sigma_e  = {theta_multi[2]:.4f}")
    print(f"  singles-only sigma_e = {theta_single[2]:.4f}")
    midpoint = 0.5 * (theta_multi[2] + theta_single[2])
    print(f"  naive count-agnostic midpoint would be ~{midpoint:.4f}")
    print("(if the balanced-count mean sigma_e sits much closer to the multis-only value "
          "than to this midpoint, that's direct evidence the pull toward a tight peak isn't "
          "just 'there happen to be more multis' -- it persists even with equal sample sizes.)")

    return sigma_es


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../data/final_kdc.csv"
    cross_evaluate_costs(csv_path)
    equal_n_subsample_test(csv_path)
