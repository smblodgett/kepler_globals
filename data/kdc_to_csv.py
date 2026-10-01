"""
Load parquet_draws/parquet_stellar/parquet_planets (written by
kdc_to_parquet.py) back out as a single CSV, selecting which columns to
include via kdc_csv_presets.py's named groups/presets rather than hand-
listing them.

parquet_draws is the big one (millions of rows); this module never loads
it all into memory at once. iter_draws_chunks()/write_csv() stream it one
manifest "part" at a time (the same row-aligned chunks kdc_to_parquet.py
wrote), merging in the small, fully-in-memory parquet_stellar/
parquet_planets tables per chunk and appending to the output CSV as they
go. load_dataframe() is a convenience wrapper around the same machinery
for when you know the selection is small enough to want back as one
in-memory DataFrame (e.g. the "minimal"/"phodymm_base" preset, or a
notebook session).

Quick examples:
    # library use
    from kdc_to_csv import write_csv
    from kdc_csv_presets import resolve_columns
    draws_cols, stellar_cols, planet_cols = resolve_columns(preset="standard")
    write_csv("csv_exports/standard.csv", draws_cols,
              stellar_columns=stellar_cols, planet_columns=planet_cols)

    # command line
    python3 kdc_to_csv.py --preset standard -o csv_exports/standard.csv
    python3 kdc_to_csv.py --draws-groups identifiers,phodymm_base \
        --catalogs KMDC,KSDC -o csv_exports/base_only.csv
    python3 kdc_to_csv.py --list-groups
    python3 kdc_to_csv.py --preset full -o csv_exports/full.csv --dry-run

See quickstart.py for a zero-argument default run.
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from kdc_to_parquet import (
    DRAWS_DIR, STELLAR_DIR, PLANETS_DIR, NULLABLE, KEY_COLS,
    read_stellar_table, read_planet_table,
)
import kdc_csv_presets as presets


# ----------------------------------------------------------------------
# Streaming draws reader
# ----------------------------------------------------------------------

def _load_draws_manifest(folder):
    folder = Path(folder)
    path = folder / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run kdc_to_parquet.py first, or pass the "
            f"right draws folder")
    return json.loads(path.read_text())


def iter_draws_chunks(columns, catalogs=None, folder=DRAWS_DIR):
    """Yield one pandas DataFrame per parquet_draws row-part (manifest
    "parts" entry), containing the manifest's key columns (kmdc_index,
    catalog) plus `columns`. Never holds more than one part in memory.

    catalogs: optional list like ["KMDC", "KSDC"] to keep only those rows;
    a part with no rows left after that filter is skipped (not yielded).
    """
    folder = Path(folder)
    m = _load_draws_manifest(folder)
    keys = m["key_columns"]
    wanted = [c for c in dict.fromkeys(columns) if c not in keys]
    unknown = [c for c in wanted if c not in m["columns"]]
    if unknown:
        raise KeyError(f"not in {folder}/manifest.json: {unknown}")

    by_group = {}
    for c in wanted:
        by_group.setdefault(m["columns"][c]["group"], []).append(c)

    if catalogs is not None:
        bad = [c for c in catalogs if c not in m["catalogs"]]
        if bad:
            raise KeyError(f"unknown catalogs {bad}; have {list(m['catalogs'])}")
        catalogs = list(catalogs)

    # every part needs at least one group's file read, to get the key
    # columns, even if `columns` resolved to nothing (e.g. KEY_COLS-only
    # request, or an all-missing preset after manifest filtering)
    if not by_group:
        by_group = {m["groups"][0]["group"]: []}

    for part in m["parts"]:
        p = part["part"]
        t = None
        for g_i, cols in by_group.items():
            path = folder / m["groups"][g_i]["files"][p]
            g_t = pq.read_table(path, columns=keys + cols)
            if t is None:
                t = g_t
            else:
                if not g_t.column(keys[0]).equals(t.column(keys[0])):
                    raise RuntimeError("row misalignment between draws column groups")
                for c in g_t.column_names:
                    if c not in t.column_names:
                        t = t.append_column(c, g_t.column(c))
        t = t.select(keys + wanted)
        if catalogs is not None:
            t = t.filter(pc.is_in(t.column("catalog"), value_set=pa.array(catalogs)))
        if t.num_rows == 0:
            continue
        yield t.to_pandas(types_mapper=NULLABLE.get)


def _resolve_tier_table(columns, loader, key, tier_dir, tier_name):
    """Load a dedup tier (stellar/planets) table and narrow it to `key` +
    `columns`. Returns None if `columns` is falsy -- the caller then skips
    attaching this tier entirely."""
    if not columns:
        return None
    df = loader(tier_dir)
    keep = [key] + [c for c in columns if c != key]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"not in {tier_name} table: {missing}")
    return df[keep]


def _estimate_csv_bytes_per_row(chunk):
    if len(chunk) == 0:
        return 0.0
    # exact for THIS chunk -- cheap since chunks are at most one manifest
    # part (bounded by TARGET_FILE_MB when kdc_to_parquet.py wrote them)
    sample = chunk if len(chunk) <= 50_000 else chunk.sample(50_000, random_state=0)
    text_len = len(sample.to_csv(index=False, header=False))
    return text_len / len(sample)


def _ordered_output_columns(draws_columns, stellar_columns, planet_columns,
                            stellar_df, planet_df):
    # KEY_COLS (kmdc_index, catalog) are the dataset's actual primary key --
    # iter_draws_chunks always reads them (every parquet_draws column group
    # carries them) but, before this, they were only used internally for
    # chunk alignment/catalogs-filtering and then silently dropped from the
    # written CSV because no preset/group ever lists them (see
    # kdc_csv_presets.py's comment on why IDENTIFIER_COLS leaves them out).
    # That's fine as a reason to skip asking the user to spell them out, but
    # not a reason to omit them from the output: without kmdc_index a CSV
    # row can't be traced back to a specific draw at all. Always lead with
    # them instead, the same way every parquet_draws column group already
    # does internally.
    cols = list(KEY_COLS) + [c for c in dict.fromkeys(draws_columns) if c not in KEY_COLS]
    if stellar_df is not None:
        if "KIC" not in cols:
            cols.append("KIC")   # the join key -- otherwise the stellar
                                  # columns can't be traced back to a star
        cols += [c for c in (stellar_columns or []) if c not in cols]
    if planet_df is not None:
        if "KOI" not in cols:
            cols.append("KOI")
        cols += [c for c in (planet_columns or []) if c not in cols]
    return cols


def _merge_tier(chunk, tier_df, on):
    """Left-merge tier_df onto chunk on `on`, after dropping any tier_df
    columns (other than `on`) that chunk already has.

    This matters because PLANET_COLS carries 'KIC' (and 'Kepler') as
    convenience columns -- real, correct, but identical to the 'KIC' a
    draws chunk already has whenever both tiers are attached together (the
    common case: KIC is in IDENTIFIER_COLS, which most presets include).
    Without this, pandas' merge() would silently rename both to
    'KIC_x'/'KIC_y' instead of raising, and out_cols' plain 'KIC' lookup
    would then fail downstream with a confusing KeyError far from the
    actual cause."""
    dupes = [c for c in tier_df.columns if c != on and c in chunk.columns]
    if dupes:
        tier_df = tier_df.drop(columns=dupes)
    return chunk.merge(tier_df, on=on, how="left")


def write_csv(output_path, draws_columns, catalogs=None,
              stellar_columns=None, planet_columns=None,
              draws_dir=DRAWS_DIR, stellar_dir=STELLAR_DIR, planet_dir=PLANETS_DIR,
              max_estimated_gb=20, force=False, verbose=True):
    """Stream parquet_draws (+ optionally parquet_stellar/parquet_planets,
    left-merged in per chunk) to a CSV at `output_path`, one manifest part
    at a time.

    draws_columns: column names from kdc_csv_presets.DRAWS_GROUPS (or any
        real draws column name). stellar_columns/planet_columns: same idea
        for the dedup tiers; leave None/[] to skip attaching that tier.
    catalogs: optional list like ["KMDC", "KSDC"] to keep only those rows.
    max_estimated_gb: after the first chunk, the resulting CSV size is
        extrapolated from that chunk's actual text size x the manifest's
        total row count; if the estimate exceeds this, write_csv raises
        instead of silently producing a huge file -- pass force=True (or
        raise the limit) once you've seen the estimate and still want it.

    Returns (output_path, rows_written).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stellar_df = _resolve_tier_table(stellar_columns, read_stellar_table, "KIC",
                                      stellar_dir, "stellar")
    planet_df = _resolve_tier_table(planet_columns, read_planet_table, "KOI",
                                     planet_dir, "planet")

    needed_draws_cols = list(dict.fromkeys(
        list(draws_columns)
        + (["KIC"] if stellar_df is not None else [])
        + (["KOI"] if planet_df is not None else [])
    ))

    manifest = _load_draws_manifest(draws_dir)
    if catalogs is None:
        total_rows = manifest["n_rows"]
    else:
        bad = [c for c in catalogs if c not in manifest["catalogs"]]
        if bad:
            raise KeyError(f"unknown catalogs {bad}; have {list(manifest['catalogs'])}")
        total_rows = sum(manifest["catalogs"][c]["row_stop"] - manifest["catalogs"][c]["row_start"]
                          for c in catalogs)

    out_cols = _ordered_output_columns(draws_columns, stellar_columns, planet_columns,
                                        stellar_df, planet_df)

    rows_written = 0
    wrote_header = False
    for chunk in iter_draws_chunks(needed_draws_cols, catalogs=catalogs, folder=draws_dir):
        if stellar_df is not None:
            chunk = _merge_tier(chunk, stellar_df, "KIC")
        if planet_df is not None:
            chunk = _merge_tier(chunk, planet_df, "KOI")
        chunk = chunk[out_cols]

        if not wrote_header:
            bytes_per_row = _estimate_csv_bytes_per_row(chunk)
            est_gb = bytes_per_row * total_rows / 1024**3
            if verbose:
                print(f"  resolved {len(out_cols)} columns x ~{total_rows:,} rows "
                      f"(estimated ~{est_gb:.2f} GB as CSV)")
            if est_gb > max_estimated_gb and not force:
                raise RuntimeError(
                    f"estimated CSV size ~{est_gb:.1f} GB exceeds max_estimated_gb="
                    f"{max_estimated_gb}. Narrow the column/catalog selection, raise "
                    f"max_estimated_gb, or pass force=True to write it anyway.")
            chunk.to_csv(output_path, index=False, mode="w", header=True)
            wrote_header = True
        else:
            chunk.to_csv(output_path, index=False, mode="a", header=False)
        rows_written += len(chunk)
        if verbose:
            print(f"  {rows_written:,}/{total_rows:,} rows written", end="\r")

    if not wrote_header:
        # every part got filtered away (e.g. an empty catalogs selection) --
        # still produce a valid, header-only CSV rather than nothing at all
        import pandas as pd
        pd.DataFrame(columns=out_cols).to_csv(output_path, index=False)

    if verbose:
        print()
        size_mb = output_path.stat().st_size / 1024**2
        print(f"  wrote {rows_written:,} rows -> {output_path} ({size_mb:.1f} MB)")
    return output_path, rows_written


def load_dataframe(draws_columns, catalogs=None, stellar_columns=None, planet_columns=None,
                    draws_dir=DRAWS_DIR, stellar_dir=STELLAR_DIR, planet_dir=PLANETS_DIR):
    """Convenience wrapper returning one in-memory DataFrame instead of
    writing a CSV -- use for a selection you know is small (a handful of
    columns, or a catalogs= filter down to one of the smaller catalogs).
    For anything close to the full dataset, prefer write_csv()."""
    import pandas as pd

    stellar_df = _resolve_tier_table(stellar_columns, read_stellar_table, "KIC",
                                      stellar_dir, "stellar")
    planet_df = _resolve_tier_table(planet_columns, read_planet_table, "KOI",
                                     planet_dir, "planet")
    needed_draws_cols = list(dict.fromkeys(
        list(draws_columns)
        + (["KIC"] if stellar_df is not None else [])
        + (["KOI"] if planet_df is not None else [])
    ))
    out_cols = _ordered_output_columns(draws_columns, stellar_columns, planet_columns,
                                        stellar_df, planet_df)

    chunks = list(iter_draws_chunks(needed_draws_cols, catalogs=catalogs, folder=draws_dir))
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=needed_draws_cols)
    if stellar_df is not None:
        df = _merge_tier(df, stellar_df, "KIC")
    if planet_df is not None:
        df = _merge_tier(df, planet_df, "KOI")
    return df[out_cols]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _csv_list(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def _build_argparser():
    ap = argparse.ArgumentParser(
        description="Load parquet_draws/parquet_stellar/parquet_planets into a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--preset", choices=list(presets.PRESETS),
                     help="named starting point from kdc_csv_presets.PRESETS")
    ap.add_argument("--draws-groups", type=_csv_list, default=[],
                     help="comma-separated kdc_csv_presets.DRAWS_GROUPS names, "
                          "layered on top of --preset")
    ap.add_argument("--stellar-groups", type=_csv_list, default=[],
                     help="comma-separated kdc_csv_presets.STELLAR_GROUPS names, "
                          "layered on top of --preset")
    ap.add_argument("--planet-groups", type=_csv_list, default=[],
                     help="comma-separated kdc_csv_presets.PLANET_GROUPS names, "
                          "layered on top of --preset")
    ap.add_argument("--extra-draws", type=_csv_list, default=[],
                     help="comma-separated explicit draws column names to add")
    ap.add_argument("--extra-stellar", type=_csv_list, default=[],
                     help="comma-separated explicit stellar column names to add")
    ap.add_argument("--extra-planet", type=_csv_list, default=[],
                     help="comma-separated explicit planet column names to add")
    ap.add_argument("--exclude-draws", type=_csv_list, default=[],
                     help="comma-separated draws column names to drop after everything "
                          "else is resolved")
    ap.add_argument("--catalogs", type=_csv_list, default=None,
                     help="comma-separated catalogs to keep, e.g. KMDC,KSDC "
                          "(default: all catalogs in the manifest)")
    ap.add_argument("-o", "--output", type=Path, default=Path("csv_exports/kdc_export.csv"))
    ap.add_argument("--draws-dir", type=Path, default=DRAWS_DIR)
    ap.add_argument("--stellar-dir", type=Path, default=STELLAR_DIR)
    ap.add_argument("--planets-dir", type=Path, default=PLANETS_DIR)
    ap.add_argument("--max-estimated-gb", type=float, default=20.0,
                     help="abort before writing if the extrapolated CSV size exceeds this")
    ap.add_argument("--force", action="store_true",
                     help="write even if the size estimate exceeds --max-estimated-gb")
    ap.add_argument("--dry-run", action="store_true",
                     help="resolve and print the column selection + size estimate, "
                          "then exit without writing")
    ap.add_argument("--list-groups", action="store_true",
                     help="print every available group/preset and exit")
    return ap


def main(argv=None):
    ap = _build_argparser()
    args = ap.parse_args(argv)

    if args.list_groups:
        print(presets.describe())
        return 0

    if args.preset is None and not args.draws_groups and not args.extra_draws:
        ap.error("pass --preset and/or --draws-groups/--extra-draws (or --list-groups "
                  "to see what's available)")

    draws_cols, stellar_cols, planet_cols = presets.resolve_columns(
        preset=args.preset,
        draws_groups=args.draws_groups,
        stellar=args.stellar_groups,
        planet=args.planet_groups,
        extra_draws=args.extra_draws,
        extra_stellar=args.extra_stellar,
        extra_planet=args.extra_planet,
        exclude_draws=args.exclude_draws,
        draws_dir=args.draws_dir, stellar_dir=args.stellar_dir, planet_dir=args.planets_dir,
    )
    if not draws_cols:
        ap.error("resolved to zero draws columns -- check --preset/--draws-groups "
                 "and --list-groups")

    print(f"draws columns ({len(draws_cols)}): {draws_cols}")
    if stellar_cols:
        print(f"stellar columns ({len(stellar_cols)}): {stellar_cols}")
    if planet_cols:
        print(f"planet columns ({len(planet_cols)}): {planet_cols}")

    if args.dry_run:
        # still want the size estimate without committing to write --
        # peek at a single chunk the same way write_csv would.
        chunk = next(iter_draws_chunks(
            list(dict.fromkeys(draws_cols
                                + (["KIC"] if stellar_cols else [])
                                + (["KOI"] if planet_cols else []))),
            catalogs=args.catalogs, folder=args.draws_dir), None)
        manifest = _load_draws_manifest(args.draws_dir)
        total_rows = manifest["n_rows"] if args.catalogs is None else sum(
            manifest["catalogs"][c]["row_stop"] - manifest["catalogs"][c]["row_start"]
            for c in args.catalogs)
        if chunk is not None:
            bpr = _estimate_csv_bytes_per_row(chunk)
            print(f"[dry-run] ~{total_rows:,} rows, estimated ~{bpr * total_rows / 1024**3:.2f} GB as CSV")
        else:
            print(f"[dry-run] 0 rows match this selection")
        return 0

    write_csv(args.output, draws_cols, catalogs=args.catalogs,
              stellar_columns=stellar_cols, planet_columns=planet_cols,
              draws_dir=args.draws_dir, stellar_dir=args.stellar_dir, planet_dir=args.planets_dir,
              max_estimated_gb=args.max_estimated_gb, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
