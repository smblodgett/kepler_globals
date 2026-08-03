"""
kg_check_ecc_degeneracy.py

Diagnostic for the point-process likelihood's eccentricity behavior.

For every unique_planet block in the flat KDC catalog (final_kdc.csv), checks how
many *distinct* eccentricity values actually appear among that planet's posterior
draws.

Why this matters: sample_eccentricity_omega() (kg_initialize_voxel_grid.py) builds
each "singles" planet's e/omega posterior via sample-importance-resampling (SIR)
from only ~1000 raw Uniform(0, 0.99) proposals, weighted by a stellar-density-
matching likelihood, then resampled WITH REPLACEMENT via rng.choice(..., p=weight).
If that weight distribution is sharply peaked (plausible: a near-circular solution
often reproduces a star's known density about as well as anything, when the data
don't strongly constrain e), the "1000 posterior draws" can collapse to a handful
of duplicated e values -- classic SIR particle degeneracy.

kg_likelihood.py's per-planet term is a log-mean-exp (soft-max-like) over exactly
these draws, so a population-level eccentricity model can get a large reward just
by placing density near wherever those degenerate duplicates happen to sit, for
many planets at once -- independent of whether that's each planet's true e. If many
"singles" collapse to a similar low-e value, that's a ready-made explanation for a
broad, radius-independent excess like the one in the eccentricity residual plot.

Usage:
    python kg_check_ecc_degeneracy.py [path/to/final_kdc.csv]
"""

import sys
import numpy as np
import pandas as pd


def check_ecc_degeneracy(csv_path, ecc_col="e", group_col="unique_planet"):
    """
    Returns one row per unique_planet with:
      - n_draws        : total posterior draws for that planet
      - n_unique_e      : number of DISTINCT eccentricity values among those draws
      - unique_frac     : n_unique_e / n_draws (near 0 => heavily degenerate)
      - dominant_e      : the single most-repeated eccentricity value
      - dominant_e_frac : fraction of that planet's draws sitting on dominant_e
      - is_single       : True for singles (unique_planet ends "_0.1"), False for
                          PhoDyMM multis -- lets you check whether degeneracy (and
                          any resulting spike) is concentrated in one population.
    """
    df = pd.read_csv(csv_path, usecols=[group_col, ecc_col])

    rows = []
    for planet_id, group in df.groupby(group_col, sort=False):
        e_vals = group[ecc_col].to_numpy()
        n_draws = len(e_vals)

        unique_vals, counts = np.unique(e_vals, return_counts=True)
        top_idx = np.argmax(counts)

        rows.append({
            "unique_planet": planet_id,
            "is_single": str(planet_id).endswith("_0.1"),
            "n_draws": n_draws,
            "n_unique_e": len(unique_vals),
            "unique_frac": len(unique_vals) / n_draws,
            "dominant_e": unique_vals[top_idx],
            "dominant_e_frac": counts[top_idx] / n_draws,
        })

    return pd.DataFrame(rows).sort_values("unique_frac").reset_index(drop=True)


def summarize(result, degenerate_thresh=0.05, spike_lo=0.0, spike_hi=0.02):
    n_total = len(result)
    degenerate = result[result["unique_frac"] < degenerate_thresh]

    print(f"n_planets total: {n_total}")
    print(f"n_planets with unique_frac < {degenerate_thresh}: {len(degenerate)} "
          f"({100 * len(degenerate) / n_total:.1f}%)")

    print("\nunique_frac summary by population:")
    print(result.groupby("is_single")["unique_frac"].describe())

    print("\n10 most degenerate planets:")
    print(result.head(10).to_string(index=False))

    if len(degenerate):
        near_spike = degenerate[degenerate["dominant_e"].between(spike_lo, spike_hi)]
        print(f"\nOf the {len(degenerate)} degenerate planets, {len(near_spike)} "
              f"have their dominant e in [{spike_lo}, {spike_hi}] "
              "(the region flagged in the residual plot).")
        print("\ndominant_e distribution among degenerate planets:")
        print(degenerate["dominant_e"].describe())


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../data/final_kdc.csv"
    result = check_ecc_degeneracy(csv_path)
    summarize(result)
    out_path = "ecc_degeneracy_report.csv"
    result.to_csv(out_path, index=False)
    print(f"\nFull per-planet report written to {out_path}")
