"""
Convert the catalog CSVs (KMDC, NCKMDC, KSDC) into THREE Parquet datasets,
split by the natural granularity of the data instead of repeating every
column on every row:

    parquet_draws/     one row per posterior draw (kmdc_index) -- the bulk
                        of the data: orbital elements, positions, derived
                        planet/system quantities that genuinely vary per
                        draw (including M_s/R_s/rho_s -- see note below).
                        Column-grouped and row-part-split exactly as before.
    parquet_planets/    one row per planet (KOI) -- Jason Rowe's per-KOI
                        transit/orbital fit columns (Period_days_rowe,
                        Rp/R*_rowe, TDur_rowe, MES_rowe, ...), which are
                        identical across every draw of a given planet and
                        were previously repeated ~1000x per planet for
                        nothing. Carries KIC and Kepler along for
                        convenience so this table joins straight to
                        parquet_systems without detouring through the draws.
    parquet_stellar/    one row per star (KIC) -- the archive/DR25 stellar-
                        catalog columns (ra/dec/kepmag/limbdark_coeff*/
                        rrmscdpp*/mesthres*/timeout*/...), stellar_source,
                        and the Rowe stellar-characterization columns
                        (Teff_rowe/R*_rowe/M*_rowe/log(g)*_rowe/rho*_rowe/
                        Z*_rowe/Source_rowe and their B-prefixed and error-
                        bar counterparts) -- all verified constant across
                        every planet of a given star (see STAR_COLS/
                        PLANET_COLS below for how that was checked), so
                        previously repeated ~2500x per star for nothing.

IMPORTANT: M_s, R_s, rho_s, c_1, c_2 stay in parquet_draws even though
they describe the star. process_singles_df resamples the star's mass and
radius ONCE PER POSTERIOR DRAW (from its own Teff/logg/rad/mass/rho
uncertainty) specifically to propagate stellar-parameter uncertainty into
each draw's own R_pE/M_pE/rho_p -- they are not redundant repeats of a
fixed value, they're real per-draw Monte Carlo realizations. Moving them
to parquet_stellar would silently discard that and make every draw look
like it used the same star.

STAR_COLS and PLANET_COLS below were derived empirically, not by guessing
from column names: every Rowe column was checked against rowe_table_final.csv
for whether it's actually constant within every multi-planet KIC (nunique()
per KIC group, dropna=False). All but Kmag_rowe are either perfectly
constant (-> STAR_COLS) or genuinely vary per planet (-> PLANET_COLS).
Kmag_rowe itself is *almost* constant per KIC (Rowe's own table wobbles by
~0.004 mag in a few multi-planet systems -- rounding noise, not signal) and
is kept in PLANET_COLS by choice: build_dedup_table's cross-check still
flags any such wobble as a warning rather than silently hiding it, in case
a future data refresh introduces a real (non-noise) difference.

Reading a subset back:
    from kdc_to_parquet import read_columns, read_stellar_table, read_planet_table
    from kdc_to_parquet import attach_stellar, attach_planets

    draws = read_columns("parquet_draws", ["M_pE", "Period_days", "hsu_flag", "KIC", "KOI"])
    draws = attach_stellar(draws)   # merges parquet_stellar's columns on KIC
    draws = attach_planets(draws)   # merges parquet_planets' columns on KOI
    stellar = read_stellar_table()  # the whole per-star table, one row per KIC
Note: pd.read_parquet("parquet_draws") on the whole folder does NOT work,
because the files hold different columns -- use read_columns. The
parquet_stellar/parquet_planets folders are each a single file, so
pd.read_parquet("parquet_stellar/kg_stellar.parquet") works directly, but
read_stellar_table()/read_planet_table() are preferred (nullable dtypes).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as pq

from kg_kmdc_col_headers import col_headers


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

INPUT_CSVS = [                      # the three catalogs -- edit as needed
    Path("thinned/KMDC.csv"),
    Path("thinned/NCKMDC.csv"),
    Path("thinned/KSDC.csv"),
]
DRAWS_DIR = Path("parquet_draws")
PLANETS_DIR = Path("parquet_planets")
STELLAR_DIR = Path("parquet_stellar")
FILE_PREFIX = "kg"

KEY_COLS = ["kmdc_index", "catalog"]   # repeated in every column group of parquet_draws
CATALOG_COL = "catalog"                # filled with each CSV's file stem
N_COLS = 10                            # data columns per group (KEY_COLS not counted)

MAX_FILE_MB = 95                    # hard limit we verify against (GitHub: 100 MB)
TARGET_FILE_MB = 70                 # what the row-part estimate aims for (headroom
                                    # for compression varying through the files)
BLOCK_SIZE = 64 << 20               # CSV bytes parsed per batch
ROW_GROUP_ROWS = 250_000            # rows buffered before each write (bigger
                                    # row groups = better compression, faster reads)
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 6


# ----------------------------------------------------------------------
# Column tiers -- see the module docstring for how these were derived
# ----------------------------------------------------------------------

_RRMSCDPP_MESTHRES_TIMEOUT_SUFFIXES = [
    "01p5", "02p0", "02p5", "03p0", "03p5", "04p5", "05p0",
    "06p0", "07p5", "09p0", "10p5", "12p0", "12p5", "15p0",
]

STELLAR_COLS = (
    [
        "ra", "dec", "kepmag",
        "limbdark_coeff1", "limbdark_coeff2", "limbdark_coeff3", "limbdark_coeff4",
        "nconfp", "nkoi", "ntce", "st_quarters",
        "dutycycle", "dutycycle_post", "dataspan", "dataspan_post",
    ]
    + [f"rrmscdpp{s}" for s in _RRMSCDPP_MESTHRES_TIMEOUT_SUFFIXES]
    + [f"mesthres{s}" for s in _RRMSCDPP_MESTHRES_TIMEOUT_SUFFIXES]
    + [f"timeout{s}" for s in _RRMSCDPP_MESTHRES_TIMEOUT_SUFFIXES]
    + ["timeoutsumry", "cdppslplong", "cdppslpshrt"]
    + ["stellar_source"]
    + [  # Rowe stellar-characterization columns -- verified constant per KIC
        "rho*_rowe", "E_rho*_rowe", "e_rho*_rowe",
        "Teff_rowe", "e_Teff_rowe",
        "R*_rowe", "E_R*_rowe", "e_R*_rowe",
        "M*_rowe", "E_M*_rowe", "e_M*_rowe",
        "log(g)*_rowe", "E_log(g)*_rowe", "e_log(g)*_rowe",
        "Z*_rowe", "e_Z*_rowe",
        "Source_rowe",
        "Brho*_rowe", "E_Brho*_rowe", "e_Brho*_rowe",
        "BTeff_rowe", "e_BTeff_rowe",
        "BR*_rowe", "E_BR*_rowe", "e_BR*_rowe",
        "BM*_rowe", "E_BM*_rowe", "e_BM*_rowe",
        "Blog(g)*_rowe", "E_Blog(g)*_rowe", "e_Blog(g)*_rowe",
        "BZ*_rowe", "e_BZ*_rowe",
    ]
)
STELLAR_KEY = "KIC"

PLANET_COLS = (
    ["KIC", "Kepler"]   # convenience columns, not used as the group key
    + [  # Jason Rowe's per-KOI transit/orbital fit columns -- verified to
        # genuinely vary per planet within a multi-planet KIC
        "Period_days_rowe", "e_Period_rowe", "T0_rowe", "e_T0_rowe",
        "Rp/R*_rowe", "E_Rp/R*_rowe", "e_Rp/R*_rowe",
        "b_rowe", "E_b_rowe", "e_b_rowe",
        "rho*M_rowe", "E_rho*M_rowe", "e_rho*M_rowe",
        "u1_rowe", "u2_rowe",
        "TTVflag_rowe", "nTTobs_rowe", "nTT_rowe",
        "TDepth_rowe", "e_TDepth_rowe",
        "TDur_rowe", "e_TDur_rowe",
        "ATDur_rowe", "e_ATDur_rowe",
        "S/N_rowe", "MES_rowe", "S/NImp_rowe",
        "chi2W_rowe", "chi2WO_rowe",
        "a/R*_rowe", "E_a/R*_rowe", "e_a/R*_rowe",
        "Inc_rowe", "E_Inc_rowe", "e_Inc_rowe",
        "Rp_rowe", "E_Rp_rowe", "e_Rp_rowe",
        "S0_rowe", "E_S0_rowe", "e_S0_rowe",
        "Kmag_rowe",            # almost-but-not-quite constant -- see docstring
        "Status_rowe",
        "BRp_rowe", "E_BRp_rowe", "e_BRp_rowe",
        "BS0_rowe", "E_BS0_rowe", "e_BS0_rowe",
    ]
)
PLANET_KEY = "KOI"

assert not (set(STELLAR_COLS) & set(PLANET_COLS)), "STELLAR_COLS/PLANET_COLS overlap"


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------

INT64 = {"kmdc_index"}
INT32 = {"KIC", "chisq_rank", "step_number", "phodymm_index"}
INT16 = {"Chain#", "nTTobs_rowe", "nTT_rowe", "nconfp", "nkoi", "ntce"}
INT8 = {"multiplicity", "Source_rowe", "stellar_source"}
BOOL = {
    "hsu_flag", "is_hidden_planet", "is_monotransiting",
    "phodymm_converged", "ecc_omega_convergence_failed",
    "timeoutsumry",  # confirmed {0.0, 1.0, NaN} only against dr25_full.csv -- a
                      # real binary flag, not a continuous quantity.
}
FLOAT64 = {
    "Period_days", "T_0", "Tp", "mean_angular_motion", "P/Pin", "P/Pout",
    "Period_days_rowe", "e_Period_rowe", "T0_rowe", "e_T0_rowe", "KOI",
}
STRING = {
    "Kepler", "TTVflag_rowe", "Status_rowe", "tm_designation",
    "teff_prov", "logg_prov", "feh_prov", "prov_sec", "st_quarters", "st_vet_date",
}

# Deliberately NOT added: dens_err1/dens_err2 (and the other archive
# _err1/_err2 columns) never reach col_headers at all -- KIC stellar-survey
# uncertainties are considered too poor to carry into the final catalog;
# where Rowe's own table doesn't have an uncertainty, none is stored.

# pandas nullable dtypes -- used both when reading parquet back out (so an
# int/bool column with nulls round-trips as Int16/boolean, not float/object)
# and inside build_dedup_table (so groupby/first/combine_first on an
# int/bool tier column don't silently upcast it to float64 along the way).
NULLABLE = {
    pa.int8(): pd.Int8Dtype(), pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(), pa.int64(): pd.Int64Dtype(),
    pa.bool_(): pd.BooleanDtype(),
}


def final_type(col):
    """Type the column is stored as in Parquet."""
    if col == CATALOG_COL: return pa.string()
    if col in INT64:   return pa.int64()
    if col in INT32:   return pa.int32()
    if col in INT16:   return pa.int16()
    if col in INT8:    return pa.int8()
    if col in BOOL:    return pa.bool_()
    if col in FLOAT64: return pa.float64()
    if col in STRING:  return pa.string()
    return pa.float32()


def read_type(col):
    """Type the CSV parser uses.

    pandas writes an int/bool column that contains NaN as floats ("3.0", "1.0",
    ""), which pyarrow's int/bool parsers reject. So parse those as float64
    (exact for integers up to 2**53) and cast afterwards in fix_types().

    st_quarters is a per-quarter bitmask string (e.g. "01111111111111111")
    that must NEVER go through this float64-then-cast path: about 22% of
    stars have a leading zero in it, which int/float parsing would silently
    strip. It's already in STRING above so final_type() returns pa.string()
    for it and this function's int/bool branch is never reached for it --
    called out here because that's easy to break by accident if STRING ever
    gets refactored.
    """
    t = final_type(col)
    if pa.types.is_integer(t) or pa.types.is_boolean(t):
        return pa.float64()
    return t


def fix_types(table):
    """Cast float64-parsed int/bool columns to their final types (nulls kept)."""
    cols = []
    for name, arr in zip(table.column_names, table.columns):
        target = final_type(name)
        if pa.types.is_boolean(target):
            bad = pc.filter(arr, pc.and_(pc.is_valid(arr),
                                         pc.invert(pc.is_in(arr, value_set=pa.array([0.0, 1.0])))))
            if len(bad):
                raise ValueError(f"{name}: non-0/1 values in bool column, e.g. "
                                 f"{bad[:5].to_pylist()}")
            arr = pc.not_equal(arr, 0.0)
        elif pa.types.is_integer(target):
            # safe cast: raises on fractional values or overflow instead of
            # silently truncating
            try:
                arr = pc.cast(arr, target, safe=True)
            except pa.ArrowInvalid as e:
                raise ValueError(f"{name} -> {target}: {e}") from None
        cols.append(arr)
    return pa.table(cols, names=table.column_names)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def csv_header(path):
    import csv as pycsv
    with open(path, newline="") as f:
        return next(pycsv.reader(f))


def parquet_bytes(table):
    """Compressed size of `table` written as Parquet, measured in memory."""
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=COMPRESSION,
                   compression_level=COMPRESSION_LEVEL)
    return sink.getvalue().size


def open_reader(csv_path, all_cols):
    """Streaming reader for one CSV + the list of wanted columns it has."""
    header = csv_header(csv_path)
    skipped = [c for c in header if c not in set(col_headers)]
    present = [c for c in all_cols if c in header]
    missing = [c for c in all_cols if c not in header and c != CATALOG_COL]
    print(f"  {csv_path}: {len(present)} columns"
          + (f", skipping {skipped}" if skipped else "")
          + (f", {len(missing)} absent -> null (e.g. {missing[:4]})" if missing else ""))
    if KEY_COLS[0] not in header:
        raise KeyError(f"{csv_path} has no {KEY_COLS[0]!r} column")
    return csv.open_csv(
        csv_path,
        read_options=csv.ReadOptions(block_size=BLOCK_SIZE),
        convert_options=csv.ConvertOptions(
            include_columns=present,          # ignore unknown/index columns
            column_types={c: read_type(c) for c in present},
            strings_can_be_null=True,         # "" in string cols -> null
            # default null_values already include "", "nan", "NaN", "NA", "null"
        ),
    )


def conform(batch, catalog, all_cols):
    """Fix types, add the catalog column, null-fill absent columns, fix order."""
    t = fix_types(pa.Table.from_batches([batch]))
    n = t.num_rows
    arrays = []
    for c in all_cols:
        if c == CATALOG_COL:
            arrays.append(pa.array([catalog] * n, pa.string()))
        elif c in t.column_names:
            arrays.append(t.column(c))
        else:
            arrays.append(pa.nulls(n, final_type(c)))
    return pa.table(arrays, names=all_cols)


def convert_draws(csv_paths):
    """The original per-draw, column-grouped, row-part-split conversion --
    unchanged except that STELLAR_COLS and the planet-only part of
    PLANET_COLS are no longer selected into it (they live in
    parquet_stellar/parquet_planets instead). KIC and KOI stay -- they're
    cheap (one float/int column each) and let a draws-only reader filter or
    group without a join.
    """
    DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    for old in DRAWS_DIR.glob(f"{FILE_PREFIX}_g*_p*.parquet"):
        old.unlink()                                  # don't leave stale parts
    print(f"=== {len(csv_paths)} CSVs -> {DRAWS_DIR}/ (per-draw)")

    draws_excluded = set(STELLAR_COLS) | (set(PLANET_COLS) - {"KIC"})

    # union of columns over all CSVs, in col_headers order
    headers = {p: set(csv_header(p)) for p in csv_paths}
    data_cols = [c for c in col_headers
                 if c not in KEY_COLS and c not in draws_excluded
                 and any(c in h for h in headers.values())]
    all_cols = KEY_COLS + data_cols
    groups = [KEY_COLS + data_cols[i:i + N_COLS]
              for i in range(0, len(data_cols), N_COLS)]

    readers = {p: open_reader(p, all_cols) for p in csv_paths}

    # first batch of EVERY csv -> worst-case bytes/row -> one rows-per-part
    firsts = {p: conform(r.read_next_batch(), p.stem, all_cols)
              for p, r in readers.items()}
    worst_bpr = max(parquet_bytes(t.select(g)) / t.num_rows
                    for t in firsts.values() for g in groups)
    rows_per_part = max(1, int(TARGET_FILE_MB * 1024**2 / worst_bpr))
    print(f"  worst group ~{worst_bpr:.1f} B/row -> up to {rows_per_part:,} rows per part")

    catalog_rows = {}                                  # stem -> [start, stop)

    def tables():
        """All CSVs in order, re-chunked into ~ROW_GROUP_ROWS-row tables."""
        nonlocal total_rows
        for p, reader in readers.items():
            catalog_rows[p.stem] = [total_rows, None]
            buf, n = [firsts[p]], firsts[p].num_rows
            for batch in reader:
                buf.append(conform(batch, p.stem, all_cols))
                n += batch.num_rows
                if n >= ROW_GROUP_ROWS:
                    yield pa.concat_tables(buf).combine_chunks()
                    buf, n = [], 0
            if buf:
                yield pa.concat_tables(buf).combine_chunks()
            catalog_rows[p.stem][1] = total_rows      # generator resumes after write

    # --- stream rows into row-aligned parts ------------------------------
    writers, part, rows_in_part, total_rows = None, -1, 0, 0
    part_rows = []                                     # [start, stop) per part
    files = {g_i: [] for g_i in range(len(groups))}

    def open_part(p):
        ws = []
        for g_i, g in enumerate(groups):
            path = DRAWS_DIR / f"{FILE_PREFIX}_g{g_i:02d}_p{p:03d}.parquet"
            schema = pa.schema([(c, final_type(c)) for c in g])
            ws.append(pq.ParquetWriter(path, schema, compression=COMPRESSION,
                                       compression_level=COMPRESSION_LEVEL))
            files[g_i].append(path.name)
        return ws

    for tbl in tables():
        offset = 0
        while offset < tbl.num_rows:
            if writers is None or rows_in_part >= rows_per_part:
                if writers is not None:
                    for w in writers:
                        w.close()
                    part_rows[-1][1] = total_rows
                part += 1
                writers = open_part(part)
                part_rows.append([total_rows, None])
                rows_in_part = 0
            n = min(rows_per_part - rows_in_part, tbl.num_rows - offset)
            chunk = tbl.slice(offset, n)
            for w, g in zip(writers, groups):
                w.write_table(chunk.select(g))
            offset += n
            rows_in_part += n
            total_rows += n
        print(f"  {total_rows:,} rows written", end="\r")

    for w in writers:
        w.close()
    part_rows[-1][1] = total_rows
    print(f"  {total_rows:,} rows, {len(groups)} groups x {part + 1} part(s)")
    for stem, (a, b) in catalog_rows.items():
        print(f"    {stem}: rows {a:,}-{b:,} ({b - a:,})")

    # --- verify sizes -----------------------------------------------------
    sizes = {p.name: p.stat().st_size / 1024**2
             for p in DRAWS_DIR.glob(f"{FILE_PREFIX}_g*_p*.parquet")}
    print(f"  largest file: {max(sizes.values()):.1f} MB")
    too_big = {k: round(v, 1) for k, v in sizes.items() if v >= MAX_FILE_MB}
    if too_big:
        raise RuntimeError(
            f"files over {MAX_FILE_MB} MB (lower TARGET_FILE_MB or N_COLS): {too_big}")

    # --- manifest ---------------------------------------------------------
    manifest = {
        "source_csvs": [p.name for p in csv_paths],
        "n_rows": total_rows,
        "key_columns": KEY_COLS,
        "catalogs": {s: {"row_start": a, "row_stop": b}
                     for s, (a, b) in catalog_rows.items()},
        "parts": [{"part": i, "row_start": a, "row_stop": b}
                  for i, (a, b) in enumerate(part_rows)],
        "groups": [{"group": g_i, "columns": g, "files": files[g_i]}
                   for g_i, g in enumerate(groups)],
        "columns": {c: {"group": g_i, "type": str(final_type(c)), "files": files[g_i]}
                    for g_i, g in enumerate(groups) for c in g if c not in KEY_COLS},
    }
    for k in KEY_COLS:
        manifest["columns"][k] = {"group": "all", "type": str(final_type(k))}
    (DRAWS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))


# ----------------------------------------------------------------------
# Dedup tiers: parquet_stellar (one row per KIC) and parquet_planets
# (one row per KOI)
# ----------------------------------------------------------------------

def build_dedup_table(name, csv_paths, group_key, tier_cols):
    """Stream csv_paths (reading only group_key + tier_cols -- the other
    ~200 draw-level columns are never parsed), reduce to one row per
    distinct group_key value, and verify every tier column really is
    constant for that key (across every row AND every catalog it appears
    in). A column that isn't perfectly constant doesn't raise -- it's
    real data, and Kmag_rowe is already known to wobble at the ~0.004 mag
    level -- but every such column is reported, with up to 5 example key
    values, so a genuine (non-noise) inconsistency introduced by a future
    data refresh doesn't silently get hidden by "first value wins".
    """
    accum = None
    mismatch_examples = {}

    def note(col, keys):
        if not len(keys):
            return
        bucket = mismatch_examples.setdefault(col, [])
        for k in keys:
            if len(bucket) >= 5:
                break
            bucket.append(k)

    for p in csv_paths:
        header = csv_header(p)
        if group_key not in header:
            print(f"  {p}: no {group_key!r} column -- skipping for {name} table")
            continue
        present = [c for c in tier_cols if c in header]
        missing = [c for c in tier_cols if c not in header]
        print(f"  {p}: {len(present)}/{len(tier_cols)} {name} columns present"
              + (f", {len(missing)} absent -> null (e.g. {missing[:4]})" if missing else ""))
        reader = csv.open_csv(
            p,
            read_options=csv.ReadOptions(block_size=BLOCK_SIZE),
            convert_options=csv.ConvertOptions(
                include_columns=[group_key] + present,
                column_types={c: read_type(c) for c in [group_key] + present},
                strings_can_be_null=True,
            ),
        )
        for batch in reader:
            t = fix_types(pa.Table.from_batches([batch]))
            df = t.to_pandas(types_mapper=NULLABLE.get)
            df = df.dropna(subset=[group_key])
            if df.empty:
                continue
            if present:
                nun = df.groupby(group_key, sort=False)[present].nunique(dropna=True)
                for c in present:
                    note(c, nun.index[nun[c] > 1].tolist())
                batch_first = df.groupby(group_key, sort=False)[present].first()
            else:
                batch_first = pd.DataFrame(index=pd.Index(df[group_key].unique(), name=group_key))
            if accum is None:
                accum = batch_first
            else:
                common = accum.index.intersection(batch_first.index)
                if len(common) and present:
                    a, b = accum.loc[common, present], batch_first.loc[common, present]
                    both_known = a.notna().values & b.notna().values
                    # a/b come from t.to_pandas(types_mapper=NULLABLE.get), so
                    # their columns are pandas nullable extension dtypes
                    # (Int64/Float64/boolean/string); .values on a DataFrame
                    # with those dtypes falls back to an object ndarray whose
                    # missing cells are the pd.NA singleton, not np.nan.
                    # pd.NA's __ne__ follows three-valued (Kleene) logic and
                    # returns pd.NA itself rather than True/False, so
                    # `a.values != b.values` is an object array that can
                    # contain actual pd.NA entries, not just True/False.
                    # `both_known & (...)` then needs numpy to evaluate
                    # bool(pd.NA) for EVERY cell (object-dtype `&` isn't
                    # lazy/short-circuited on the left operand), which raises
                    # "boolean value of NA is ambiguous" even for cells where
                    # both_known is False and the NA result would've been
                    # discarded anyway. np.not_equal's `where=` makes the
                    # comparison itself skip any cell both_known already
                    # marks False, so pd.NA is never compared there at all.
                    differ = np.zeros_like(both_known, dtype=bool)
                    np.not_equal(a.values, b.values, where=both_known, out=differ)
                    for j, c in enumerate(present):
                        note(c, common[differ[:, j]].tolist())
                accum = accum.combine_first(batch_first)
                accum.index.name = group_key

    if accum is None:
        accum = pd.DataFrame(columns=tier_cols)
        accum.index.name = group_key
    for c in tier_cols:
        if c not in accum.columns:
            accum[c] = pd.array([None] * len(accum), dtype="object")
    accum = accum[tier_cols]

    if mismatch_examples:
        print(f"[warn] {name} table: not perfectly constant per {group_key} for these "
              f"columns (kept the first value seen -- check whether this is measurement "
              f"noise, like Kmag_rowe's known ~0.004 mag wobble, or a real signal that "
              f"belongs in a finer-grained tier):")
        for c, keys in mismatch_examples.items():
            print(f"    {c}: e.g. {group_key} in {keys}")

    return accum.reset_index()


def convert_dedup_tier(name, out_dir, csv_paths, group_key, tier_cols):
    print(f"=== {len(csv_paths)} CSVs -> {out_dir}/ ({group_key}-level)")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{FILE_PREFIX}_{name}.parquet"
    out_path.unlink(missing_ok=True)

    df = build_dedup_table(name, csv_paths, group_key, tier_cols)
    table = pa.Table.from_pandas(df, preserve_index=False)
    # from_pandas already infers correct arrow types from the nullable pandas
    # dtypes build_dedup_table used, but run it through the same schema
    # final_type() promises everywhere else, in case a fully-null tier column
    # (e.g. every catalog missing it) came back as arrow null-type instead.
    table = table.cast(pa.schema([(group_key, final_type(group_key))]
                                  + [(c, final_type(c)) for c in tier_cols]))
    pq.write_table(table, out_path, compression=COMPRESSION,
                   compression_level=COMPRESSION_LEVEL)

    size_mb = out_path.stat().st_size / 1024**2
    print(f"  {table.num_rows:,} distinct {group_key} value(s), "
          f"{len(tier_cols)} columns, {size_mb:.1f} MB")
    if size_mb >= MAX_FILE_MB:
        raise RuntimeError(
            f"{out_path} is {size_mb:.1f} MB (>= {MAX_FILE_MB}); this tier was assumed "
            f"small enough for one file -- ask for row-splitting to be added here")

    manifest = {
        "source_csvs": [p.name for p in csv_paths],
        "group_key": group_key,
        "n_rows": table.num_rows,
        "file": out_path.name,
        "columns": {c: str(final_type(c)) for c in [group_key] + tier_cols},
    }
    (out_dir / f"manifest_{name}.json").write_text(json.dumps(manifest, indent=1))


# ----------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------

def read_columns(folder, columns, catalogs=None, to_pandas=True):
    """Read `columns` (plus the key columns) from a parquet_draws-shaped
    folder (column-grouped, row-part-split -- NOT parquet_stellar or
    parquet_planets, which are single files; use read_stellar_table /
    read_planet_table for those).

    catalogs: optional list like ["KMDC", "KSDC"] to keep only those rows.
    """
    folder = Path(folder)
    m = json.loads((folder / "manifest.json").read_text())
    keys = m["key_columns"]
    wanted = [c for c in dict.fromkeys(columns) if c not in keys]
    unknown = [c for c in wanted if c not in m["columns"]]
    if unknown:
        raise KeyError(f"not in {folder}: {unknown}")
    by_group = {}
    for c in wanted:
        by_group.setdefault(m["columns"][c]["group"], []).append(c)
    if not by_group:
        by_group = {0: []}

    # only open the parts that overlap the requested catalogs
    parts = range(len(m["parts"]))
    if catalogs is not None:
        bad = [c for c in catalogs if c not in m["catalogs"]]
        if bad:
            raise KeyError(f"unknown catalogs {bad}; have {list(m['catalogs'])}")
        spans = [m["catalogs"][c] for c in catalogs]
        parts = [p["part"] for p in m["parts"]
                 if any(p["row_start"] < s["row_stop"] and s["row_start"] < p["row_stop"]
                        for s in spans)]

    out = None
    for g_i, cols in by_group.items():
        files = [folder / m["groups"][g_i]["files"][p] for p in parts]
        t = pa.concat_tables(pq.read_table(f, columns=keys + cols) for f in files)
        if out is None:
            out = t
        else:
            if not t.column(keys[0]).equals(out.column(keys[0])):
                raise RuntimeError("row misalignment between groups")
            for c in cols:
                out = out.append_column(c, t.column(c))
    out = out.select(keys + wanted)
    if catalogs is not None:
        out = out.filter(pc.is_in(out.column("catalog"), value_set=pa.array(catalogs)))
    if not to_pandas:
        return out
    return out.to_pandas(types_mapper=NULLABLE.get)


def _read_dedup_table(out_dir, name):
    m = json.loads((out_dir / f"manifest_{name}.json").read_text())
    t = pq.read_table(out_dir / m["file"])
    return t.to_pandas(types_mapper=NULLABLE.get)


def read_stellar_table(folder=STELLAR_DIR):
    """The whole per-star table (one row per KIC)."""
    return _read_dedup_table(Path(folder), "stellar")


def read_planet_table(folder=PLANETS_DIR):
    """The whole per-planet table (one row per KOI)."""
    return _read_dedup_table(Path(folder), "planets")


def attach_stellar(df, folder=STELLAR_DIR, columns=None):
    """Left-merge parquet_stellar's columns onto `df` (must have a 'KIC'
    column) on KIC. `columns`: optional subset of STELLAR_COLS to bring in
    (default: all of them). Including 'KIC' in `columns` is harmless -- it
    won't be duplicated."""
    stellar = read_stellar_table(folder)
    if columns is not None:
        stellar = stellar[["KIC"] + [c for c in columns if c != "KIC"]]
    return df.merge(stellar, on="KIC", how="left")


def attach_planets(df, folder=PLANETS_DIR, columns=None):
    """Left-merge parquet_planets' columns onto `df` (must have a 'KOI'
    column) on KOI. `columns`: optional subset of PLANET_COLS to bring in
    (default: all of them, including the convenience KIC/Kepler columns --
    pass e.g. columns=['MES_rowe'] to skip those). Including 'KOI' in
    `columns` is harmless -- it won't be duplicated."""
    planets = read_planet_table(folder)
    if columns is not None:
        planets = planets[["KOI"] + [c for c in columns if c != "KOI"]]
    return df.merge(planets, on="KOI", how="left")


if __name__ == "__main__":
    found = [p for p in INPUT_CSVS if p.exists()]
    for p in INPUT_CSVS:
        if not p.exists():
            print(f"!! {p} not found, skipping")
    if found:
        convert_draws(found)
        convert_dedup_tier("stellar", STELLAR_DIR, found, STELLAR_KEY, STELLAR_COLS)
        convert_dedup_tier("planets", PLANETS_DIR, found, PLANET_KEY, PLANET_COLS)
