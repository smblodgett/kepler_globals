import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.csv as ar_csv
from mpi4py import MPI

from pathlib import Path
import sys
import os

from kg_random_row_selector import PHODYMM_PATH, find_koi

sys.path.append(str(Path.cwd().parent / "src"))
from kg_initialize_voxel_grid import process_singles_df
from kg_constants import *

# All ranks read/filter the catalogs and call process_singles_df() together --
# it is a collective MPI operation (scatter/gather) that every rank must enter.
# Only rank 0 gets back a real DataFrame (every other rank gets None), so only
# rank 0 should go on to merge/derive columns/save -- see the rank==0 guard below.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


stellar_data_filename = "../data/berger_2020_keplerstellar.tsv"
rowe_stellar_data_filename ="../data/rowe_table_final.csv"
additional_stellar_data_filename = "keplerstellar.csv"

dr_25_data_filename = "../data/q1_q17_dr25.csv"


def find_converged_systems():
    kois = []
    for folder in os.listdir(PHODYMM_PATH):
            folder_path = os.path.join(PHODYMM_PATH, folder)
            if os.path.isdir(folder_path):
                for path in os.listdir(folder_path):
                    path_full = os.path.join(folder_path, path)
                    if os.path.isdir(path_full) and path == 'analysis_dir':
                        for file in os.listdir(path_full):
                            if file == 'dqa_allparam.csv':
                                path_full = os.path.join(path_full, file)
                                print(path_full)
                                koi = find_koi(path_full)
                                kois.append(koi)
    return kois


def rowe_table_attach_multi(ncmultis_dr_df, df):
    """
    Attaches Jason Rowe's table (rowe_table_final.csv) to df: KIC, KOI,
    Kepler, and every *_rowe column, plus 'is_monotransiting'.

    Adapted from kg_subsampler.py's rowe_table_attach(), which processes one
    koi (one star) at a time: it grabs every Rowe row for that koi's KOI
    number, then matches each of the system's planets to one of those rows
    by period proximity (+/- 0.2 days) using the MEAN of that planet's
    resampled posterior periods, with a fallback for mono-transiting planets
    (Jason Rowe's negative-Period_days_rowe sentinel for a single-transit
    candidate whose period isn't really constrained).

    This version differs in three ways, all because df here holds every
    non-converged multi system and every Monte Carlo draw at once instead of
    one system's posterior at a time:

      1. Matching is grouped by 'kepid' (== Rowe's own 'KIC' column) instead
         of looping over one koi's rows at a time.
      2. Each planet is matched using its DR25 CATALOG period
         (ncmultis_dr_df's 'period' column), not the mean of 1000 resampled
         posterior periods -- a planet's physical identity, and therefore
         which Rowe row belongs to it, doesn't change from draw to draw, so
         there's nothing to gain (and 1000x more work to do) matching per
         draw, or averaging over noisy resampled periods, when the catalog
         value is sitting right there.
      3. The match is done ONCE per (kepid, planet_number) and then
         broadcast onto that planet's 1000 draws with a merge, instead of a
         python loop writing into a boolean-masked df.loc slice per planet.
         As a side effect this also tracks which Rowe rows have already been
         claimed within a system (kg_subsampler's version doesn't), so two
         planets with close periods can't both match the same Rowe row.
    """
    rowe_df = pd.read_csv(rowe_stellar_data_filename, engine='pyarrow')

    # Every Rowe column except 'KIC' itself -- df already has 'KIC' from the
    # stellar_df merge above, and merging in a second copy would just get
    # suffixed (KIC_x/KIC_y) instead of lining up.
    rowe_value_cols = [c for c in rowe_df.columns if c not in ("KIC", "Unnamed: 0")]

    # One row per physical planet, with its catalog period -- what we match
    # against Rowe's table. (Every one of a planet's 1000 draws shares the
    # same catalog period, so drop_duplicates leaves exactly one row per
    # (kepid, planet_number).)
    planets = (
        ncmultis_dr_df[['kepid', 'planet_number', 'period']]
        .drop_duplicates(['kepid', 'planet_number'])
        .reset_index(drop=True)
    )
    for col in rowe_value_cols:
        planets[col] = None  # object dtype -- several Rowe columns are strings (Kepler name, Source_rowe, ...)

    unmatched = []
    for kepid, planet_group in planets.groupby('kepid'):
        kic_rows = rowe_df[rowe_df['KIC'] == kepid]
        if kic_rows.empty:
            unmatched.extend(planet_group.index.tolist())
            continue

        used = pd.Series(False, index=kic_rows.index)
        still_unmatched = []

        for idx, planet_row in planet_group.iterrows():
            period = planet_row['period']
            mask = (
                (kic_rows['Period_days_rowe'] > period - 0.2) &
                (kic_rows['Period_days_rowe'] < period + 0.2) &
                (~used)
            )
            candidates = kic_rows.loc[mask]
            if len(candidates) == 0:
                still_unmatched.append(idx)
                continue
            # More than one candidate within the window -- take the closest
            # by period rather than leave it ambiguous.
            best = (candidates['Period_days_rowe'] - period).abs().idxmin()
            used.loc[best] = True
            planets.loc[idx, rowe_value_cols] = kic_rows.loc[best, rowe_value_cols].values

        # Mono-transiting fallback, same rule kg_subsampler's rowe_table_attach
        # uses: only take it when it's unambiguous (exactly one leftover
        # planet and exactly one leftover mono-transit candidate for this
        # kepid) -- otherwise there's no period information left to say which
        # unmatched planet goes with which mono-transit candidate.
        mono_candidates = kic_rows.loc[(kic_rows['Period_days_rowe'] < 0) & (~used)]
        if len(still_unmatched) == 1 and len(mono_candidates) == 1:
            idx = still_unmatched[0]
            best = mono_candidates.index[0]
            planets.loc[idx, rowe_value_cols] = kic_rows.loc[best, rowe_value_cols].values
            still_unmatched = []
        elif still_unmatched and len(mono_candidates) > 0:
            print(
                f"kepid {kepid}: {len(still_unmatched)} planet(s) unmatched by period and "
                f"{len(mono_candidates)} mono-transiting Rowe candidate(s) available -- "
                f"ambiguous with no period to disambiguate by, leaving all unmatched."
            )

        unmatched.extend(still_unmatched)

    for idx in unmatched:
        row = planets.loc[idx]
        print(f"No Rowe table match found for kepid {row['kepid']} planet_number {row['planet_number']}; leaving its Rowe columns blank.")

    # Flag mono-transiting matches the same way kg_subsampler does: whichever
    # planet ended up attached to a Rowe row with a negative Period_days_rowe
    # (the "period unconstrained" sentinel). Unmatched planets (still None)
    # are left un-flagged, not assumed one way or the other.
    planets["is_monotransiting"] = (pd.to_numeric(planets["Period_days_rowe"], errors="coerce") < 0).astype(int)

    df = df.merge(planets.drop(columns=['period']), on=['kepid', 'planet_number'], how='left')
    return df


def main():
    stellar_df = pd.read_csv(stellar_data_filename,engine='pyarrow',delimiter='\t') # used to be from ../data/keplerstellar.csv, now is from Berger et al 2020

    print("len(stellar_df) before cuts: ",len(stellar_df))

    # Make the cuts to stellar catalog based off of temperature, logg
    # stellar_df = stellar_df[(stellar_df["Teff"]>4000) & (stellar_df["Teff"]<7000)]
    # stellar_df = stellar_df[(stellar_df["logg"]>4)]

    additional_stellar_df = pd.read_csv(additional_stellar_data_filename,engine='pyarrow')

    stellar_df = stellar_df.merge(
                                additional_stellar_df,
                                left_on='KIC',
                                right_on='KIC',
                                how='left'

                            )

    # Read in the expanded stellar df, which has CDPP values
    rowe_stellar_df = pd.read_csv(rowe_stellar_data_filename,engine='pyarrow') # this is the stellar data from Rowe et al 2015.
    # rowe_stellar_df = rowe_stellar_df[rowe_stellar_df["st_delivname"]=="q1_q17_dr25_stellar"]

    rowe_stellar_df['multiplicity'] = rowe_stellar_df['KIC'].map(rowe_stellar_df['KIC'].value_counts())

    rowe_multis = rowe_stellar_df[rowe_stellar_df['multiplicity']>1]

    rowe_multis['koi_prad'] = rowe_multis['Rp_rowe']

    rowe_multis['koi_prad_err1'] = rowe_multis['e_Rp_rowe']

    rowe_multis['koi_prad_err2'] = rowe_multis['E_Rp_rowe']

    rowe_multis['koi_prad_err2'] = rowe_multis['E_Rp_rowe']

    rowe_multis['koi_period'] = rowe_multis['Period_days_rowe']

    rowe_multis['koi_period_err1'] = rowe_multis['e_Period_rowe']

    rowe_multis['koi_period_err2'] = rowe_multis['e_Period_rowe']

    rowe_multis['koi_impact'] = rowe_multis['b_rowe']

    rowe_multis['koi_impact_err1'] = rowe_multis['b_rowe_e']

    rowe_multis['koi_impact_err2'] = rowe_multis['b_rowe_E']

    rowe_multis['koi_duration'] = rowe_multis['TDur_rowe']

    rowe_multis['koi_duration_err1'] = rowe_multis['e_TDur_rowe']

    rowe_multis['koi_duration_err2'] = rowe_multis['e_TDur_rowe']





    # This used to merge Jason Rowe's table (rowe_table_final.csv) onto
    # stellar_df here, keyed on 'KIC' alone. That table has up to 8 rows per
    # KIC (one per Kepler candidate planet, not one per star), so a plain
    # merge('KIC') duplicated stellar_df's rows for every multi-planet star --
    # and that duplication would have cascaded into df below, via
    # processed_ncmultis_dr_df.merge(stellar_df, ...). Jason Rowe's columns
    # are attached properly, per PLANET, by rowe_table_attach_multi() further
    # down instead (see its docstring).
    dr_df = pd.read_csv(dr_25_data_filename,engine='pyarrow')

    # create a column for the multipliity of each DR25 planet
    dr_df['multiplicity'] = dr_df['kepid'].map(dr_df['kepid'].value_counts())

    converged_multis_kois = find_converged_systems()

    multis_dr_df = dr_df[dr_df["multiplicity"]!=1] 

    ncmultis_dr_df = multis_dr_df[~multis_dr_df["kepid"].isin(converged_multis_kois)]

    rowe_multis = rowe_multis[~rowe_multis["KIC"].isin(converged_multis_kois)]

    print("len of ncmultis_dr_df: ",len(ncmultis_dr_df)) 

    print("len of rowe_multis: ",len(ncmultis_dr_df)) 


    print("len of multis_dr_df: ",len(multis_dr_df))

    print("nonconverged multi kois: ", ncmultis_dr_df["kepid"].tolist())



    print("sum ncmultis_dr_df['kepid'].isin(stellar_df['KIC']) : ", np.sum(ncmultis_dr_df["kepid"].isin(stellar_df['KIC'])))

    ncmultis_dr_df = ncmultis_dr_df[ncmultis_dr_df["kepid"].isin(ncmultis_dr_df['KIC'])]
    print("sum ncmultis_dr_df : ", np.sum(ncmultis_dr_df["kepid"]))



    print("ncmultis before removal of bad period error: ")
    # Remove the planets in the nc multis df that have nans in their period errors, since we need these for sampling the posteriors
    ncmultis_dr_df = ncmultis_dr_df[~(ncmultis_dr_df["koi_period_err1"].isna() | ncmultis_dr_df["koi_period_err2"].isna())]
    print("ncmultis after removal of bad period errors: ")



    # Reset the index so we can iterate through nc multis df
    ncmultis_dr_df = ncmultis_dr_df.reset_index(drop=True)

    ncmultis_dr_df = ncmultis_dr_df.sort_values(['kepid', 'koi_period'])

    ncmultis_dr_df['planet_number'] = (
        ncmultis_dr_df.groupby('kepid').cumcount() + 1
        )    
# Give the singles df the same cols as the multis df, sample ecc and omega for the singles
    processed_ncmultis_dr_df = process_singles_df(ncmultis_dr_df,stellar_df,0.01,10,seed=333,validation_graph=False,make_graphs=False)

    # process_singles_df() draws num_posteriors_per_planet=1000 rows per input
    # planet, in the same order as ncmultis_dr_df's own rows -- but its output
    # columns are only ["R_pE","Period_days","M_pE","e","omega","i","R_s","M_s","kepid"];
    # 'planet_number' doesn't come back. Restore it here: output block i (rows
    # i*1000 .. i*1000+999) belongs to ncmultis_dr_df's i-th row, so repeating
    # each planet_number 1000 times lines it back up. find_crossing_planets()
    # right below already assumes this column exists on its input.
    processed_ncmultis_dr_df['planet_number'] = np.repeat(ncmultis_dr_df['planet_number'].values, 1000)


    def find_crossing_planets(processed_ncmultis_dr_df):
        # np.tile(arange(1000), N) makes an array of length 1000*N -- we need
        # length N (one draw label per row), i.e. len(df)//1000 repeats of the
        # 0..999 cycle, since process_singles_df() lays out its output as one
        # block of num_posteriors_per_planet=1000 consecutive draws per planet.
        processed_ncmultis_dr_df["draw"] = np.tile(np.arange(1000), len(processed_ncmultis_dr_df) // 1000)

        processed_ncmultis_dr_df['a'] = (G * (processed_ncmultis_dr_df['M_s'] * MSKG + processed_ncmultis_dr_df['M_pE'] * MEKG) * (processed_ncmultis_dr_df['Period_days'] * 24 * 3600)**2 / (4 * np.pi**2) )**(1/3)

        processed_ncmultis_dr_df['q'] = processed_ncmultis_dr_df['a'] * (1 - processed_ncmultis_dr_df['e'])
        processed_ncmultis_dr_df['Q'] = processed_ncmultis_dr_df['a'] * (1 + processed_ncmultis_dr_df['e'])

        processed_ncmultis_dr_df = processed_ncmultis_dr_df.sort_values(['kepid','draw','a'])

        inner_Q = processed_ncmultis_dr_df.groupby(['kepid','draw'])['Q'].shift(1)
        crosses_inner = inner_Q > processed_ncmultis_dr_df['q']

        outer_q = processed_ncmultis_dr_df.groupby(['kepid','draw'])['q'].shift(-1)
        crosses_outer = outer_q < processed_ncmultis_dr_df['Q']

        orbit_crosses = crosses_inner | crosses_outer

        crossing_planets = (
            processed_ncmultis_dr_df.loc[orbit_crosses]
            .groupby('kepid')['planet_number']
            .apply(list)
        )

        return crossing_planets.to_dict()

    crossing_planets = find_crossing_planets(processed_ncmultis_dr_df)
    

    i = 0
    ncmultis_dr_df_copy = ncmultis_dr_df.copy()
    reprocessed = pd.DataFrame(columns=processed_ncmultis_dr_df.columns)
    # crossing_planets is a plain dict (find_crossing_planets returns
    # .to_dict()) -- dict has no .empty() method (that's a DataFrame/Series
    # thing), so this raised AttributeError before the loop ever ran. An empty
    # dict is already falsy, so a plain truthiness check is both correct and
    # simpler.
    while crossing_planets:
        crossing_pairs = (
                        pd.Series(crossing_planets)
                        .explode()
                        .rename_axis('kepid')
                        .rename('planet_number')
                        .reset_index()
                    )

        ncmultis_dr_df_copy = ncmultis_dr_df_copy.merge(
            crossing_pairs,
            on=['kepid', 'planet_number'],
            how='inner'
        )

        reprocessed_0 = process_singles_df(ncmultis_dr_df_copy,stellar_df,0.01,10,seed=333,validation_graph=False,make_graphs=False)
        # Same fix as above: process_singles_df() drops 'planet_number', and
        # find_crossing_planets(reprocessed_0) below needs it.
        reprocessed_0['planet_number'] = np.repeat(ncmultis_dr_df_copy['planet_number'].values, 1000)
        crossing_planets = find_crossing_planets(reprocessed_0)
        reprocessed = pd.concat([reprocessed,reprocessed_0])


        if i > 20:
            print("Warning: more than 20 iterations of orbit crossing removal. Stopping.")
            break
        i+=1


    processed_ncmultis_dr_df = pd.concat([processed_ncmultis_dr_df,reprocessed])

    # Re-derive planet_number per DRAW, not once per kepid: sort by draw
    # BEFORE Period_days, and group the cumcount() by ['kepid', 'draw'] so it
    # restarts at 1 for every draw. ('period' isn't a column on this df --
    # process_singles_df's output column is 'Period_days'; sorting by
    # ['kepid', 'period', 'draw'] would also have raised a KeyError.)
    processed_ncmultis_dr_df = processed_ncmultis_dr_df.sort_values(['kepid', 'draw', 'Period_days'])

    processed_ncmultis_dr_df['planet_number'] = (
        processed_ncmultis_dr_df.groupby(['kepid', 'draw']).cumcount() + 1
        )

    # Every row is now already uniquely identified by (kepid, draw,
    # planet_number) -- there's nothing left to group here. (What this
    # replaces, .groupby('kepid','planet_number','draw'), was broken two
    # ways: groupby()'s group keys go in as ONE argument, a list -- three
    # bare positional args get parsed as by='kepid', axis='planet_number',
    # ... which raises a TypeError; and even as
    # .groupby(['kepid','planet_number','draw']) with nothing aggregated
    # after it, that reassigns processed_ncmultis_dr_df to a
    # DataFrameGroupBy, not a DataFrame -- and the very next statement calls
    # .merge() on it, which DataFrameGroupBy doesn't have.)

       

    if rank == 0:
        print("finished processing!")

        df = processed_ncmultis_dr_df.merge(
                                                                stellar_df,
                                                                left_on='kepid',
                                                                right_on='KIC',
                                                                how='left'
                                                        )

        # process_singles_df() only returns ["R_pE","Period_days","M_pE","e","omega","kepid"] --
        # it consumes koi_impact/koi_duration (and their error columns) internally but never
        # carries them through. Bring them back from the DR25 catalog so b_trans/T_total_hr
        # below can resample from the raw catalog values.
        #
        # ncmultis_dr_df has one row per PLANET, not one row per star -- unlike the
        # singles-only version this was adapted from (where kepid alone was a safe
        # merge key because multiplicity == 1 meant exactly one row per kepid).
        # Merging on 'kepid' alone here would match every planet in df against ALL
        # of that kepid's rows in ncmultis_dr_df, cross-joining a 3-planet system's
        # rows 3-for-1 and handing some planets another planet's koi_impact/
        # koi_duration. 'planet_number' has to be part of the join key too.
        df = df.merge(
                        ncmultis_dr_df[['kepid', 'planet_number', 'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
                                        'koi_duration', 'koi_duration_err1', 'koi_duration_err2']],
                        on=['kepid', 'planet_number'],
                        how='left'
                    )

        print("finished merging!")

        df['M_s'] = df['Mass']
        df['R_s'] = df['Rad']
        df['c_1'] = np.nan
        df['c_2'] = np.nan
        df['R_p/R_s'] = df['R_pE'] * RETORS / df['R_s']
        df['R_pJ'] = df['R_pE'] * RJTORE
        df['rho_p'] = df['M_pE'] * MEG / (4/3 * np.pi * (df['R_pE'] * RECM)**3)
        df['rho_s'] = 10**(df['rho']) * RHOS
        df['M_p/M_s'] = df['M_pE'] * MEKG / (df['M_s'] * MSKG)
        df['M_pJ'] = df['M_pE'] * METOMJ
        df['sqrt(e)_cos(omega)'] = np.sqrt(df['e']) * np.cos(df['omega'] * np.pi / 180)
        df['sqrt(e)_sin(omega)'] = np.sqrt(df['e']) * np.sin(df['omega'] * np.pi / 180)

        df['b_trans'] = np.random.normal(df['koi_impact'], np.max(np.abs([df['koi_impact_err1'], df['koi_impact_err2']]), axis=0), size=len(df))

        df['Omega'] = 0
        df['is_hidden_planet'] = 0
        df['planet'] = 0

        ## orbital angles
        df['true_anomaly'] = (90 - df['omega']) % 360
        df['eccentric_anomaly'] = ((180 / np.pi) * np.arctan2((np.sqrt(1-df['e']**2)*np.sin(df['true_anomaly']*np.pi/180)),(df['e']+np.cos(df['true_anomaly']*np.pi/180)))) % 360
        df['mean_anomaly'] = ((180 / np.pi) * ((np.pi / 180 ) * df['eccentric_anomaly']) - (df['e']*np.sin(df['eccentric_anomaly']*np.pi/180))) % 360 # M, the mean anomaly (19 degrees for KOI 500.01)
        df['mean_longitude'] = (df['Omega'] + df['omega'] + df['mean_anomaly']) % 360 # mean longitude of planet at epoch ::: longitude of ascending node (always 0 for our system) + argument of periapse (little omega) + mean anomaly (always close to 90 degrees)


        ## orbital distances
        df['a_AU'] = ((df['Period_days']*DTOS)**2 * G * ((df['M_s']*MSKG) + (df['M_pE']*MEKG))/(4*np.pi**2))**(1/3) * MTOAU # semimajor axis in AU
        df['a_R_s'] = (df['a_AU']/RSAU) / df['R_s'] # semimajor axis in stellar radii
        df['peri_AU'] = df['a_AU'] * (1 - df['e']) # periastron in AU
        df['peri_R_s'] = (df['peri_AU']/RSAU) / df['R_s'] # periastron in stellar radii
        df['apo_AU'] = df['a_AU'] * (1 + df['e']) # apoastron in AU
        df['apo_R_s'] = (df['apo_AU']/RSAU) / df['R_s'] # apoastron in stellar radii
        df['d_AU'] = df['a_AU']*(1 - df['e']**2) / (1 + (df['e']*np.cos(df['true_anomaly']*np.pi/180))) # star-planet separation at transit in AU
        df['d_R_s'] = (df['d_AU']/RSAU) / df['R_s'] # star-planet separation at transit in stellar radii

        ## impact, probability, and duration parameters
        # b_trans is an independent normal draw (line above) and a_R_s is derived separately
        # from Period/M_s/M_pE/R_s, so cos(i) = b_trans*R_s/a_R_s isn't guaranteed to land in
        # [-1, 1] -- an unlucky sample can push it just outside, which makes arccos silently
        # return NaN (with a RuntimeWarning) instead of raising. Clip into the valid domain,
        # but log how many rows needed it and by how much: a few hits at ~1e-10 are just
        # floating-point noise, while many rows or a large excess means b_trans and a_R_s are
        # systematically inconsistent for those planets and is worth investigating separately.
        cos_i = df['b_trans'] * df['R_s'] / df['a_R_s']
        n_invalid = int((cos_i.abs() > 1).sum())
        if n_invalid:
            max_excess = float((cos_i.abs() - 1).clip(lower=0).max())
            print(f"[warn] {n_invalid}/{len(df)} rows have |b_trans*R_s/a_R_s| > 1 "
                f"(max excess {max_excess:.3g}); clipping to the arccos domain [-1, 1]")
        df['i'] = np.arccos(cos_i.clip(-1, 1)) * 180 / np.pi

        df['b_occ'] = (df['a_R_s'] * np.cos(df['i']*np.pi/180)) * ((1-df['e']**2)/(1-df['e']*np.sin(df['omega']*np.pi/180))) # occultation impact parameter
        df['p_trans'] = ((df['R_s'] * RSAU + df['R_pJ']*RJAU) / df['a_AU']) * ((1+df['e']*np.sin(df['omega']*np.pi/180)) / (1-df['e']**2)) # transit probability
        df['p_occ'] = ((df['R_s'] * RSAU + df['R_pJ']*RJAU) / df['a_AU']) * ((1-df['e']*np.sin(df['omega']*np.pi/180)) / (1-df['e']**2)) # occultation probability

        # df['T_total_hr'] = 24 * (df['Period_days'] / np.pi) * np.arcsin((df['R_s']*RSAU/df['a_AU'])*(np.sqrt((1+ df['R_p/R_s'])**2 - df['b_trans']**2)/np.sin(df['i']*np.pi/180))) * ((np.sqrt(1-df['e']**2))/(1+df['e']*np.sin(df['omega']*np.pi/180))) # total duration of transit (t4 - t1)
        df['T_total_hr'] = np.random.normal(df['koi_duration'] * 24, np.max(np.abs([df['koi_duration_err1'], df['koi_duration_err2']]), axis=0), size=len(df)) # total duration of transit (t4 - t1) from DR25

        df['T_full_hr'] = 24 * (df['Period_days'] / np.pi) * np.arcsin((df['R_s']*RSAU/df['a_AU'])*(np.sqrt(np.maximum(0,(1-df['R_p/R_s'])**2 - df['b_trans']**2))/np.sin(df['i']*np.pi/180))) * ((np.sqrt(1-df['e']**2))/(1+df['e']*np.sin(df['omega']*np.pi/180))) # full duration of transit (t3 - t2)
        df['K_RV'] = (2*np.pi*G/(df['Period_days']*24*60*60))**(1/3) * ((MSKG*df['M_pJ']*np.sin(df['i']*np.pi/180)/MSTOMJ)/((df['M_s']*MSKG)+(MSKG*df['M_pJ']/MSTOMJ))**(2/3)) * (1/(1-df['e']**2)**(1/2))  # amplitude of radial velocity variations    ## make sure units are right here. should be m/s

        from kg_subsampler import occurrence_rate_params, is_in_hsu

        df = occurrence_rate_params(df)
        df = is_in_hsu(df)  # sets 'hsu_flag': whether KIC is in the Hsu et al. stellar catalog
        df = rowe_table_attach_multi(ncmultis_dr_df, df)  # attaches Jason Rowe's table (KOI, Kepler, *_rowe columns) and sets 'is_monotransiting' -- see the function's docstring for how this differs from kg_subsampler.py's per-system rowe_table_attach()



        df["P/Pin"] = -1
        df["P/Pout"] = -1
        df["Tdur/Tdurin"] = -1
        df["Tdur/Tdurout"] = -1
        df["R/Rin"] = -1
        df["R/Rout"] = -1
        df["M/Min"] = -1
        df["M/Mout"] = -1
        df["rho/rhoin"] = -1
        df["rho/rhoout"] = -1
        df["i-iin"] = -1
        df["iout-i"] = -1
        df["xiin"] = -1
        df["xiout"] = -1
        df["distin_hillrad"] = -1
        df["distout_hillrad"] = -1
        df["distin_hillrad_e"] = -1
        df["distout_hillrad_e"] = -1
        df["e/ein"] = -1
        df["eout/e"] = -1
        df["omega-omegain"] = -1
        df["omegaout-omega"] = -1

        # ------------------------------------------------------------------
        # Pairwise (inner/outer neighbor) comparison columns -- translated
        # from kg_subsampler.py's system_params(). That version loops over
        # df.index.unique() (one index value == one system-draw, since it
        # processes one PhoDyMM system at a time) and picks out the
        # inner/outer planet via PhoDyMM's decimal planet numbering
        # (planet - 0.1 / planet + 0.1).
        #
        # This df instead holds every non-converged multi system and every
        # Monte Carlo draw all at once, so "one system-draw" here is a
        # ('kepid', 'draw') pair rather than a single index value, and
        # planets are identified by integer 'planet_number' (1, 2, 3, ...,
        # assigned via groupby('kepid').cumcount() near the top of this
        # file, in order of increasing period) instead of PhoDyMM's
        # x.1/x.2/... scheme.
        #
        # find_crossing_planets() above already assumes 'planet_number' and
        # 'draw' exist on this df (and uses this same
        # groupby(['kepid','draw'])[...].shift() idiom to compare a planet
        # to its neighbor by semimajor axis) -- both columns need to
        # survive the process_singles_df()/crossing-removal steps above for
        # this to work. Fail loudly here rather than silently producing
        # all -1 columns if they don't.
        missing = {"kepid", "draw", "planet_number"} - set(df.columns)
        assert not missing, (
            f"pairwise comparison columns need {missing} on df -- make sure "
            "they survive the process_singles_df()/crossing-removal steps above"
        )

        # 'planet_number' is a fixed identity per planet (like PhoDyMM's
        # decimal numbering), not something re-derived per draw -- but the
        # shift(1)/shift(-1) below only picks up the correct inner/outer
        # neighbor if rows within each ('kepid', 'draw') group are actually
        # ordered by planet_number. Sort once up front rather than assume it.
        df = df.sort_values(["kepid", "draw", "planet_number"]).reset_index(drop=True)
        sys_draw = df.groupby(["kepid", "draw"])

        is_first = df["planet_number"] == sys_draw["planet_number"].transform("min")
        is_last = df["planet_number"] == sys_draw["planet_number"].transform("max")

        period, t_dur, radius, mass, density = df["Period_days"], df["T_total_hr"], df["R_pE"], df["M_pE"], df["rho_p"]
        inc, omega, ecc, sm_axis, mass_ratio = df["i"], df["omega"], df["e"], df["a_AU"], df["M_p/M_s"]

        period_in, period_out = sys_draw["Period_days"].shift(1), sys_draw["Period_days"].shift(-1)
        t_dur_in, t_dur_out = sys_draw["T_total_hr"].shift(1), sys_draw["T_total_hr"].shift(-1)
        radius_in, radius_out = sys_draw["R_pE"].shift(1), sys_draw["R_pE"].shift(-1)
        mass_in, mass_out = sys_draw["M_pE"].shift(1), sys_draw["M_pE"].shift(-1)
        density_in, density_out = sys_draw["rho_p"].shift(1), sys_draw["rho_p"].shift(-1)
        inc_in, inc_out = sys_draw["i"].shift(1), sys_draw["i"].shift(-1)
        omega_in, omega_out = sys_draw["omega"].shift(1), sys_draw["omega"].shift(-1)
        ecc_in, ecc_out = sys_draw["e"].shift(1), sys_draw["e"].shift(-1)
        sm_axis_in, sm_axis_out = sys_draw["a_AU"].shift(1), sys_draw["a_AU"].shift(-1)
        mass_ratio_in, mass_ratio_out = sys_draw["M_p/M_s"].shift(1), sys_draw["M_p/M_s"].shift(-1)

        # -- inner-neighbor (planet_number - 1) comparisons --
        df["P/Pin"] = np.where(~is_first, period / period_in, -1)
        df["Tdur/Tdurin"] = np.where(~is_first, t_dur / t_dur_in, -1)
        df["R/Rin"] = np.where(~is_first, radius / radius_in, -1)
        df["M/Min"] = np.where(~is_first, mass / mass_in, -1)
        df["rho/rhoin"] = np.where(~is_first, density / density_in, -1)
        df["i-iin"] = np.where(~is_first, (inc - inc_in) % 360, -1)
        df["omega-omegain"] = np.where(~is_first, (omega - omega_in) % 360, -1)
        df["e/ein"] = np.where(~is_first, ecc / ecc_in, -1)
        df["xiin"] = np.where(~is_first, (t_dur_in / t_dur) * (period / period_in) ** (1 / 3), -1)
        df["distin_hillrad"] = np.where(
            ~is_first,
            (sm_axis - sm_axis_in) / (((mass_ratio + mass_ratio_in) / 3) ** (1 / 3) * ((sm_axis + sm_axis_in) / 2)),
            -1,
        )
        df["distin_hillrad_e"] = np.where(
            ~is_first,
            ((sm_axis * (1 - ecc)) - (sm_axis_in * (1 + ecc_in))) / (((mass_ratio + mass_ratio_in) / 3) ** (1 / 3) * ((sm_axis + sm_axis_in) / 2)),
            -1,
        )

        # -- outer-neighbor (planet_number + 1) comparisons --
        df["P/Pout"] = np.where(~is_last, period / period_out, -1)
        df["Tdur/Tdurout"] = np.where(~is_last, t_dur / t_dur_out, -1)
        df["R/Rout"] = np.where(~is_last, radius / radius_out, -1)
        df["M/Mout"] = np.where(~is_last, mass / mass_out, -1)
        df["rho/rhoout"] = np.where(~is_last, density / density_out, -1)
        df["iout-i"] = np.where(~is_last, (inc_out - inc) % 360, -1)
        df["omegaout-omega"] = np.where(~is_last, (omega_out - omega) % 360, -1)
        df["eout/e"] = np.where(~is_last, ecc_out / ecc, -1)
        df["xiout"] = np.where(~is_last, (t_dur / t_dur_out) * (period_out / period) ** (1 / 3), -1)
        df["distout_hillrad"] = np.where(
            ~is_last,
            (sm_axis_out - sm_axis) / (((mass_ratio + mass_ratio_out) / 3) ** (1 / 3) * ((sm_axis + sm_axis_out) / 2)),
            -1,
        )
        df["distout_hillrad_e"] = np.where(
            ~is_last,
            ((sm_axis_out * (1 - ecc_out)) - (sm_axis * (1 + ecc))) / (((mass_ratio + mass_ratio_out) / 3) ** (1 / 3) * ((sm_axis + sm_axis_out) / 2)),
            -1,
        )


        df["dilute"] = -1
        df["chisq"] = -1
        df["Chain#"] = np.nan
        df["chisq_rank"] = np.nan
        df["step_number"] = np.nan
        df["phodymm_index"] = np.nan
        df["phodymm_converged"] = -1
        df["is_in_DR25"] = np.nan



        df["omega_rad"] = df["omega"] * np.pi / 180
        df['falsetrueanomaly'] = ((np.pi/2) - df['omega_rad']) % (2*np.pi)

        # find the true anomaly
        df['f'] = ((np.pi/2)
                    - df['omega_rad']
                    - (df['e'] * np.cos(df['omega_rad']) * np.cos(df['i']*np.pi/180)**2 / (1+df['e']*np.sin(df['omega_rad'])))) % (2*np.pi)

        # find eccentric anomaly
        df['eccentric_anomaly_hamann'] = (np.arctan2(np.sqrt(1-df['e']**2)*np.sin(df['f']),df['e']+np.cos(df['f']))) % (2*np.pi)
        df['false_eccentric_anomaly'] = (np.arctan2(np.sqrt(1-df['e']**2)*np.sin(df['falsetrueanomaly']),df['e']+np.cos(df['falsetrueanomaly']))) % (2*np.pi)

        # find mean anomaly
        df['mean_anomaly_hamann'] = (df['eccentric_anomaly_hamann'] - (df['e']*np.sin(df['eccentric_anomaly_hamann']))) % (2*np.pi)
        df['false_mean_anomaly'] = (df['false_eccentric_anomaly'] - (df['e']*np.sin(df['false_eccentric_anomaly']))) % (2*np.pi)

        df['mean_angular_motion'] = 2*np.pi/ df['Period_days']
        df["mean_anomaly_hamann_800"] = np.nan
        df["mean_anomaly_hamann_850"] = np.nan
        df["corrected_mean_anomaly_800"] = np.nan
        df["eccentric_anomaly_hamann_800"] = np.nan
        df["eccentric_anomaly_hamann_850"] = np.nan
        df["true_anomaly_hamann_800"] = np.nan
        df["true_anomaly_hamann_850"] = np.nan
        df["corrected_eccentric_anomaly_800"] = np.nan
        df["corrected_true_anomaly_800"] = np.nan

        df["interior_mass_pJ"] = 0


        df = df.sort_values(['kepid', 'draw', 'planet_number'])

        # Cumulative sum of planet mass within each (kepid, draw) group, ordered by
        # planet_number (== ordered by period, innermost to outermost). shift(1) so
        # each planet gets the sum of planets *interior* to it, not including itself.
        df['interior_mass_pJ'] = (
            df.groupby(['kepid', 'draw'])['M_pJ']
            .transform(lambda x: x.cumsum().shift(1, fill_value=0))
        )


        df["mu"] = (
                GAU * (
                    df['M_s']
                + df['M_pJ']    / MSTOMJ
                + df['interior_mass_pJ'] / MSTOMJ
                )
            )
        df['q'] = df['a_AU'] * (1 - df['e'])
        df["Tp"] = np.nan
        df["x"] = np.nan
        df["y"] = np.nan
        df["z"] = np.nan
        df["vx"] = np.nan
        df["vy"] = np.nan
        df["vz"] = np.nan
        df["T_0"] = np.nan



        # kg_subsampler.py uses 'chisq_rank' here (a small per-system PhoDyMM
        # chain rank) as the last digits of kmdc_index. That doesn't exist for
        # these non-converged, non-PhoDyMM-fit systems, so use the draw number
        # instead (+1 so it's 1-indexed, running 1..1000) -- it plays the same
        # role (a per-row-within-planet id), and 1000 still zfills to exactly 4
        # digits, unlike the previous df-wide M_pE rank, which could run past
        # 4 digits and overflow the fixed-width id.
        id_number_identifier = df["draw"] + 1
        koi_parts = df["KOI"].astype(str).str.split(".", n=1, expand=True).reindex(columns=[0, 1])
        real_kmdc_index = (
            koi_parts[0].str.zfill(4)                          # XXXX padded
            + koi_parts[1]                                     # YY
            + id_number_identifier.astype(str).str.zfill(4)    # Z padded
        )
        df['kmdc_index'] = real_kmdc_index

        df[df['Period_days_rowe'] != np.nan]['is_in_rowe'] = 1
        df[df['Period_days_rowe'] != 1]['is_in_rowe'] = 0


        df[df['KOI'].isin(multis_dr_df['kepoi_name'].str.replace('K', '').astype(float))]['is_in_DR25'] = 1
        df[df['is_in_DR25'] != 1]['is_in_DR25'] = 0




        from kg_kmdc_col_headers import col_headers
        df = df[col_headers]

        table = pa.Table.from_pandas(df)
        ar_csv.write_csv(table, f"thinned/nckmdc.csv")

        print(f"Saved ksdc")


if __name__ == "__main__":
    main()
