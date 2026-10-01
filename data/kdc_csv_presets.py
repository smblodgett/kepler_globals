"""
Column menu for kdc_to_csv.py: names groups of parquet_draws/parquet_stellar/
parquet_planets columns by what they ARE (raw PhoDyMM chain output vs.
something kg_subsampler.py/create_ksdc.py/create_nckmdc.py computed from
that output), plus a few ready-made PRESETS that combine them for common
use cases.

Where each group's membership comes from (not guessed from column names):
  - PHODYMM_BASE_COLS is exactly the raw per-draw MCMC chain columns
    kg_subsampler.py's column_rename() renames (the star's 4-wide tail
    block: M_s/R_s/c_1/c_2, and each planet's repeating 9-wide block:
    planet/Period_days/T_0/sqrt(e)_cos(omega)/sqrt(e)_sin(omega)/i/Omega/
    M_pJ/R_p/R_s), plus chisq, Chain#, and dilute -- fit statistic, chain
    id, and dilution nuisance parameter respectively, all read straight off
    the chain file with no renaming needed, so column_rename() never
    touches them, but they're just as "raw" as the renamed nine.
  - DRAWS_DERIVED_* are exactly what kg_subsampler.py's calculate_params(),
    mean_anomaly_corrections(), and add_interior_mass_and_positions()
    compute FROM those base columns (split into three groups because
    DERIVED_ANOMALY_COLS in particular is rarely needed and roughly
    doubles the column count on its own).
  - PERTURBATION_RATIO_COLS is the crossing-planet dynamical-context family
    (P/Pin, i-iin, distin_hillrad, ...) -- useful for stability/crossing
    analyses specifically, skippable otherwise.
  - IDENTIFIER_COLS is row/system bookkeeping (which chain, which step,
    which planet slot, how many planets in the system) -- not physics, but
    you probably always want it.
  - FLAG_COLS is the small set of boolean/categorical quality/provenance
    flags calculate_params()/process_singles_df set per draw.
  - STELLAR_* and PLANET_* are derived from kdc_to_parquet.py's own
    STELLAR_COLS/PLANET_COLS (the already-verified-constant Rowe/archive
    column lists used to build parquet_stellar/parquet_planets), split
    further here into "_rowe" vs. archive/DR25 columns so you can pull in
    just one flavor of stellar characterization. Importing them rather
    than re-listing them means this file can never drift out of sync with
    what's actually in those two parquet tables.

Everything here is just names -- resolve_columns() is what checks a
preset/group selection against a REAL manifest.json (or parquet_draws'
column list for stellar/planets) before handing columns to kdc_to_csv.py,
so a column that's missing from a particular run (not every run has every
catalog) is dropped with a warning instead of crashing kdc_to_csv.py.
"""

from kdc_to_parquet import (
    STELLAR_COLS, PLANET_COLS, DRAWS_DIR, STELLAR_DIR, PLANETS_DIR,
)
import json
from pathlib import Path


# ----------------------------------------------------------------------
# parquet_draws column groups
# ----------------------------------------------------------------------

IDENTIFIER_COLS = [
    # kmdc_index/catalog are the manifest's own key_columns and are always
    # included by read_columns/iter_draws_chunks regardless of what you ask
    # for, so they're left out of this list on purpose -- listing them here
    # too would just make every column count reported by --list-groups /
    # --dry-run look one-too-many for no reason.
    "KIC", "KOI",               # also usable as join keys into the stellar/
                                # planet tiers without attaching them
    "planet",                   # which member of the system this row is
    "multiplicity",             # planet count for this row's system -- a
                                # count, not a measurement, but bookkeeping
                                # you want on nearly every row
    "Chain#", "step_number", "phodymm_index",  # chain id / row position,
                                                # straight off the raw file
]

PHODYMM_BASE_COLS = [
    # the star's 4-wide tail block
    "M_s", "R_s", "c_1", "c_2",
    # each planet's repeating 9-wide block (minus 'planet', already above)
    "Period_days", "T_0",
    "sqrt(e)_cos(omega)", "sqrt(e)_sin(omega)",
    "i", "Omega", "M_pJ", "R_p/R_s",
    # raw, unrenamed chain columns
    "dilute", "chisq",
]

DERIVED_CORE_COLS = [
    # calculate_params(): masses/radii/densities, angles, orbital distances,
    # impact/probability/duration -- the columns almost every analysis of
    # this catalog actually wants
    "R_pJ", "R_pE", "M_pE", "rho_p", "rho_s", "M_p/M_s",
    "e", "omega",
    "a_AU", "a_R_s", "peri_AU", "apo_AU",
    "b_trans", "b_occ", "p_trans", "p_occ",
    "T_total_hr", "T_full_hr", "K_RV",
]

DERIVED_ANOMALY_COLS = [
    # calculate_params()'s true/eccentric/mean anomaly chain, plus
    # mean_anomaly_corrections()'s whole Hamann-et-al re-derivation of the
    # same quantities at fixed epochs (800/850 BKJD) -- rarely needed
    # outside anomaly-specific work, and this alone is ~20 columns
    "true_anomaly", "eccentric_anomaly", "mean_anomaly", "mean_longitude",
    "omega_rad", "falsetrueanomaly", "f",
    "eccentric_anomaly_hamann", "false_eccentric_anomaly",
    "mean_anomaly_hamann", "false_mean_anomaly", "mean_angular_motion",
    "mean_anomaly_hamann_800", "mean_anomaly_hamann_850",
    "corrected_mean_anomaly_800",
    "eccentric_anomaly_hamann_800", "eccentric_anomaly_hamann_850",
    "true_anomaly_hamann_800", "true_anomaly_hamann_850",
    "corrected_eccentric_anomaly_800", "corrected_true_anomaly_800",
]

DERIVED_POSITION_COLS = [
    # add_interior_mass_and_positions(): interior-mass-weighted mu/time of
    # pericenter/Jacobian Cartesian state vectors, plus the two
    # peri/apo-in-stellar-radii and separation-at-transit columns
    # calculate_params() computes alongside them
    "interior_mass_pJ", "mu", "q", "Tp",
    "x", "y", "z", "vx", "vy", "vz",
    "peri_R_s", "apo_R_s", "d_AU", "d_R_s",
]

PERTURBATION_RATIO_COLS = [
    # crossing-planet dynamical context (this planet vs. its inner/outer
    # neighbor) -- a stability/crossing-analysis group, not general-purpose
    "P/Pin", "P/Pout", "Tdur/Tdurin", "Tdur/Tdurout",
    "R/Rin", "R/Rout", "M/Min", "M/Mout", "rho/rhoin", "rho/rhoout",
    "i-iin", "iout-i", "xiin", "xiout",
    "distin_hillrad", "distout_hillrad", "distin_hillrad_e", "distout_hillrad_e",
    "e/ein", "eout/e", "omega-omegain", "omegaout-omega",
]

OCCURRENCE_RATE_COLS = ["occurrence_rate_hsu", "E_or_hsu", "e_or_hsu"]

FLAG_COLS = [
    "hsu_flag", "is_hidden_planet", "is_monotransiting",
    "phodymm_converged", "ecc_omega_convergence_failed",
    "completeness", "chisq_rank",
]

DRAWS_GROUPS = {
    "identifiers": IDENTIFIER_COLS,
    "phodymm_base": PHODYMM_BASE_COLS,
    "derived_core": DERIVED_CORE_COLS,
    "derived_anomaly": DERIVED_ANOMALY_COLS,
    "derived_position": DERIVED_POSITION_COLS,
    "perturbation_ratios": PERTURBATION_RATIO_COLS,
    "occurrence_rate": OCCURRENCE_RATE_COLS,
    "flags": FLAG_COLS,
}
DRAWS_ALL_COLS = [c for g in DRAWS_GROUPS.values() for c in g]
assert len(DRAWS_ALL_COLS) == len(set(DRAWS_ALL_COLS)), "a column is listed in >1 draws group"


# ----------------------------------------------------------------------
# parquet_stellar / parquet_planets column groups
# ----------------------------------------------------------------------

ROWE_STELLAR_COLS = [c for c in STELLAR_COLS if c.endswith("_rowe")]
STELLAR_FLAG_COLS = [c for c in STELLAR_COLS if c == "stellar_source"]
ARCHIVE_STELLAR_COLS = [c for c in STELLAR_COLS
                        if c not in ROWE_STELLAR_COLS and c not in STELLAR_FLAG_COLS]
assert set(ROWE_STELLAR_COLS) | set(STELLAR_FLAG_COLS) | set(ARCHIVE_STELLAR_COLS) == set(STELLAR_COLS)

STELLAR_GROUPS = {
    "stellar_archive": ARCHIVE_STELLAR_COLS,   # DR25/archive catalog columns
    "stellar_rowe": ROWE_STELLAR_COLS,         # Rowe stellar characterization
    "stellar_flags": STELLAR_FLAG_COLS,
    "stellar_all": list(STELLAR_COLS),
}

PLANET_ID_COLS = [c for c in PLANET_COLS if c in ("KIC", "Kepler")]
ROWE_PLANET_FIT_COLS = [c for c in PLANET_COLS if c not in PLANET_ID_COLS]

PLANET_GROUPS = {
    "planet_ids": PLANET_ID_COLS,              # convenience KIC/Kepler cols
    "planet_rowe": ROWE_PLANET_FIT_COLS,       # Rowe per-planet transit/orbital fit
    "planet_all": list(PLANET_COLS),
}

# A small, curated subset for the "standard"/quickstart preset -- not every
# Rowe column, just the ones most often wanted alongside the KMDC's own
# per-draw fit.
STANDARD_STELLAR_COLS = ["Teff_rowe", "R*_rowe", "M*_rowe", "kepmag"]
STANDARD_PLANET_COLS = ["Period_days_rowe", "T0_rowe", "Rp_rowe", "TDur_rowe", "MES_rowe"]


# ----------------------------------------------------------------------
# Presets: ready-made combinations for the common cases
# ----------------------------------------------------------------------
# Each preset is (draws_groups, stellar_groups_or_explicit_cols, planet_groups_or_explicit_cols).
# stellar/planet entries can be a list of STELLAR_GROUPS/PLANET_GROUPS names
# ("group:...") or an explicit column list ("cols:...") -- see resolve_columns().

PRESETS = {
    "minimal": {
        "draws_groups": ["identifiers", "phodymm_base"],
        "stellar": [],
        "planet": [],
    },
    "phodymm_base": {   # alias of "minimal" under the name you asked for
        "draws_groups": ["identifiers", "phodymm_base"],
        "stellar": [],
        "planet": [],
    },
    "standard": {
        "draws_groups": ["identifiers", "phodymm_base", "derived_core", "flags"],
        "stellar": [("cols", STANDARD_STELLAR_COLS)],
        "planet": [("cols", STANDARD_PLANET_COLS)],
    },
    "full": {
        "draws_groups": list(DRAWS_GROUPS),
        "stellar": [("group", "stellar_all")],
        "planet": [("group", "planet_all")],
    },
}


# ----------------------------------------------------------------------
# Resolving a selection against a live manifest
# ----------------------------------------------------------------------

def _manifest_columns(draws_dir):
    path = Path(draws_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run kdc_to_parquet.py first (or pass the "
            f"right --draws-dir) before selecting draws columns")
    return set(json.loads(path.read_text())["columns"])


def _dedup_table_columns(tier_dir, manifest_name):
    path = Path(tier_dir) / f"manifest_{manifest_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run kdc_to_parquet.py first (or pass the "
            f"right directory) before selecting these columns")
    return set(json.loads(path.read_text())["columns"])


def _expand(names, groups, kind):
    """names: list of group-name strings. Unknown names raise with the
    available choices listed, so a typo doesn't silently resolve to nothing."""
    out = []
    bad = [n for n in names if n not in groups]
    if bad:
        raise KeyError(f"unknown {kind} group(s) {bad}; have {list(groups)}")
    for n in names:
        out.extend(groups[n])
    return out


def _expand_tier(entries, groups, kind):
    """entries: list of ("group", name) or ("cols", [explicit, column, names])."""
    out = []
    for kind_tag, value in entries:
        if kind_tag == "group":
            out.extend(_expand([value], groups, kind))
        elif kind_tag == "cols":
            out.extend(value)
        else:
            raise ValueError(f"tier entry must be ('group', name) or ('cols', [...]); got {kind_tag!r}")
    return out


def resolve_columns(preset=None, draws_groups=None, stellar=None, planet=None,
                     extra_draws=None, extra_stellar=None, extra_planet=None,
                     exclude_draws=None,
                     draws_dir=DRAWS_DIR, stellar_dir=STELLAR_DIR, planet_dir=PLANETS_DIR,
                     verbose=True):
    """Turn a preset name and/or explicit group selections into three column
    lists (draws, stellar, planet), each filtered down to columns that
    actually exist in the manifest at `draws_dir`/`stellar_dir`/`planet_dir`
    for THIS run (a column missing from one run's manifest -- not every
    catalog has every column -- is dropped with a warning, not an error).

    preset: a name from PRESETS, or None.
    draws_groups: additional DRAWS_GROUPS names, layered on top of the
        preset's (or used alone if preset is None).
    stellar / planet: additional STELLAR_GROUPS/PLANET_GROUPS names
        (plain strings), layered on top of the preset's.
    extra_draws / extra_stellar / extra_planet: explicit column names,
        layered on top of everything else (for the one-off column the
        group taxonomy doesn't happen to cover).
    exclude_draws: column names to drop after everything else is resolved
        (handy for e.g. "standard" minus the anomaly columns someone added
        back in by hand).

    Returns (draws_cols, stellar_cols, planet_cols) -- stellar_cols/
    planet_cols are None (not []) when nothing was requested for that tier,
    which is what kdc_to_csv.write_csv()/load_dataframe() use to decide
    whether to attach that tier at all.
    """
    draws_names, stellar_entries, planet_entries = [], [], []
    if preset is not None:
        if preset not in PRESETS:
            raise KeyError(f"unknown preset {preset!r}; have {list(PRESETS)}")
        p = PRESETS[preset]
        draws_names += list(p["draws_groups"])
        stellar_entries += list(p["stellar"])
        planet_entries += list(p["planet"])
    draws_names += list(draws_groups or [])
    stellar_entries += [("group", g) for g in (stellar or [])]
    planet_entries += [("group", g) for g in (planet or [])]

    draws_cols = list(dict.fromkeys(
        _expand(draws_names, DRAWS_GROUPS, "draws") + list(extra_draws or [])))
    stellar_cols = list(dict.fromkeys(
        _expand_tier(stellar_entries, STELLAR_GROUPS, "stellar") + list(extra_stellar or [])))
    planet_cols = list(dict.fromkeys(
        _expand_tier(planet_entries, PLANET_GROUPS, "planet") + list(extra_planet or [])))

    draws_cols = [c for c in draws_cols if c not in set(exclude_draws or [])]

    # --- filter against the live manifest(s) ---
    available = _manifest_columns(draws_dir)
    missing = [c for c in draws_cols if c not in available]
    if missing and verbose:
        print(f"[kdc_csv_presets] dropping {len(missing)} draws column(s) not "
              f"present in {draws_dir}/manifest.json: {missing}")
    draws_cols = [c for c in draws_cols if c in available]

    if stellar_cols:
        available = _dedup_table_columns(stellar_dir, "stellar") | {"KIC"}
        missing = [c for c in stellar_cols if c not in available]
        if missing and verbose:
            print(f"[kdc_csv_presets] dropping {len(missing)} stellar column(s) not "
                  f"present in {stellar_dir}/manifest_stellar.json: {missing}")
        stellar_cols = [c for c in stellar_cols if c in available]

    if planet_cols:
        available = _dedup_table_columns(planet_dir, "planets") | {"KOI"}
        missing = [c for c in planet_cols if c not in available]
        if missing and verbose:
            print(f"[kdc_csv_presets] dropping {len(missing)} planet column(s) not "
                  f"present in {planet_dir}/manifest_planets.json: {missing}")
        planet_cols = [c for c in planet_cols if c in available]

    return draws_cols, (stellar_cols or None), (planet_cols or None)


def describe():
    """Human-readable listing of every group/preset and its columns, for
    `kdc_to_csv.py --list-groups`."""
    lines = ["=== draws groups ==="]
    for name, cols in DRAWS_GROUPS.items():
        lines.append(f"  {name} ({len(cols)}): {', '.join(cols)}")
    lines.append("=== stellar groups ===")
    for name, cols in STELLAR_GROUPS.items():
        lines.append(f"  {name} ({len(cols)}): {', '.join(cols)}")
    lines.append("=== planet groups ===")
    for name, cols in PLANET_GROUPS.items():
        lines.append(f"  {name} ({len(cols)}): {', '.join(cols)}")
    lines.append("=== presets ===")
    for name, p in PRESETS.items():
        lines.append(f"  {name}: draws_groups={p['draws_groups']}, "
                      f"stellar={p['stellar']}, planet={p['planet']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
