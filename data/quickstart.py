"""
Zero-argument default export: parquet_draws/parquet_stellar/parquet_planets
-> one CSV, using kdc_csv_presets.PRESETS["standard"] (identifiers +
PhoDyMM's own raw per-draw fit parameters + calculate_params()'s core
derived columns + the quality/provenance flags, plus a small curated
subset of Rowe's stellar and per-planet columns) -- not "minimal" (that
skips R_pE/M_pE/e/a_AU/etc., which almost everyone wants) and not "full"
(every column in every tier -- genuinely huge; see kdc_to_csv.py --preset
full if you actually want that).

Run as-is for a sane default:
    python3 quickstart.py
or override just the output path / which catalogs to include:
    python3 quickstart.py --output csv_exports/mine.csv --catalogs KSDC

Anything more specific than that (different preset, extra columns, a
tighter/looser size cap, ...) -- use kdc_to_csv.py directly; run
    python3 kdc_to_csv.py --list-groups
to see every option this could instead be built from.
"""

import argparse
from pathlib import Path

from kdc_to_csv import write_csv
from kdc_csv_presets import resolve_columns, DRAWS_DIR, STELLAR_DIR, PLANETS_DIR


DEFAULT_OUTPUT = Path("csv_exports/kdc_quickstart.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--catalogs", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                     default=None, help="comma-separated catalogs to keep, e.g. KMDC,KSDC "
                                        "(default: all catalogs in the manifest)")
    ap.add_argument("--draws-dir", type=Path, default=DRAWS_DIR)
    ap.add_argument("--stellar-dir", type=Path, default=STELLAR_DIR)
    ap.add_argument("--planets-dir", type=Path, default=PLANETS_DIR)
    args = ap.parse_args(argv)

    print("quickstart: using the 'standard' preset (see kdc_csv_presets.PRESETS['standard'])")
    draws_cols, stellar_cols, planet_cols = resolve_columns(
        preset="standard",
        draws_dir=args.draws_dir, stellar_dir=args.stellar_dir, planet_dir=args.planets_dir,
    )
    write_csv(args.output, draws_cols, catalogs=args.catalogs,
              stellar_columns=stellar_cols, planet_columns=planet_cols,
              draws_dir=args.draws_dir, stellar_dir=args.stellar_dir, planet_dir=args.planets_dir)
    print(f"done -- {args.output}")
    print("want more/fewer columns? see: python3 kdc_to_csv.py --list-groups")


if __name__ == "__main__":
    main()
