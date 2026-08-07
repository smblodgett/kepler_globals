'''
kg_subsampler.py
================

Processes the output data of a PhoDyMM planetary system fit.

This module takes raw converged PhoDym posterior spaces
which have been randomly subsampled by the kg_random_row_selector
script and creates a dataframe, one per system, then calculates or
appends a variety of extra parameters, including orbital elements,
occurrence rates, system comparisons, and the entire table of
Lissauer et al. 2024. Finally, it appends the dataframe into a large
combined csv with all planets, thinned/all_thin.csv.

Usage
-----
Run from the command line:
    python kg_subsampler.py koi_number

Parameters
----------
koi_number : int, optional
    Specifies which koi system the subsampler will run on. If no
    koi is specified, the subsampler will run on all available systems.

Author
------
Steven Blodgett <blodgett.steven.m@gmail.com>
Created on: 2024-11-11
'''


import pandas as pd
import numpy as np
from math import fmod
from sorcha.ephemeris.orbit_conversion_utilities import universal_cartesian
import os
import re
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from kg_constants import *  # Import constants from kg_constants.py
from kg_kmdc_col_headers import col_headers

RAW_PATH = '/hdd2/backup/danielkj/PhoDyMM_results_final/completed_systems/'   # pathway to directory with raw PhoDyMM output posterior data
SUBSAMPLED_PATH = '/home/byu.local/smb9564/research/hierarchal_modeling/kepler_globals/data/subsampled_rows/' # pathway to directory containing lists of subsampled rows for each different KOI output by PhoDyMM
SAVE_PATH = '/home/byu.local/smb9564/research/hierarchal_modeling/kepler_globals/data/thinned/'  # pathway to directory to save data

_CSV_CACHE = {}  # avoids re-reading the same reference CSV from disk on every call


def _cached_read_csv(path, **kwargs):
    """
    pd.read_csv, cached by path for the life of the process.

    occurrence_rate_params/rowe_table_attach/is_in_hsu each get called once per
    planet or once per system, and each call previously re-read its reference
    CSV from disk from scratch -- including hsu_stellar_catalog_output.csv
    (~19 MB, 80006 rows), re-read once per system (661x total across a full
    run) and occurrence_rates_hsu.csv, re-read once per PLANET (~1665x total).
    None of these files are ever mutated in place after being read here, so
    it's safe to hand back the same cached DataFrame to every caller.
    """
    if path not in _CSV_CACHE:
        _CSV_CACHE[path] = pd.read_csv(path, **kwargs)
    return _CSV_CACHE[path]




def find_koi(name_with_numbers):
    """Grab the koi number from filename and return it as a string."""
    numbers = re.findall(r'\d+', name_with_numbers)
    result = ''.join(numbers)
    return result


def process_dataframe(df,koi):
    """
    Process PhoDyMM dataframe into a more usable form.

    This includes changing the column header names, appending other tables into the dataframe,
    and calculating a variety of derived values (e.g. planetary density).

    Parameters
    ----------
    df : DataFrame
      The dataframe derived from one PhoDyMM output posterior.
    koi : str
      The koi number of the system (there will often be a leading 0).

    Returns
    -------
    final_system_df : DataFrame
      The final processed dataframe.
    """
    step_number_col = df['Unnamed: 0']
    phodymm_index_col = df['phodymm_index'] # PhoDyMM's raw file row position (see make_df_from_subsample) -- distinct from step_number, which is per-chain and resets/repeats across chains

    df = df.drop(['Unnamed: 0', 'phodymm_index'], axis=1)
    total_columns = len(df.columns)

    n_chunks = (total_columns - 7) // 9

    processed_chunks = []

    # Iterate through each chunk (9 columns) before the last 7 columns.
    for i in range(n_chunks):
        chunk_columns = df.iloc[:, i*9:(i+1)*9]
        constant_columns = df.iloc[:, -7:]
        chunk_df = pd.concat([chunk_columns, constant_columns], axis=1)
        chunk_df = column_rename(chunk_df)
        chunk_df['chisq_rank'] = chunk_df['chisq'].rank(method='min', ascending=True) # chisq ranking
        chunk_df['step_number'] = step_number_col # PhoDyMM step number (per-chain, resets/repeats across chains)
        chunk_df['phodymm_index'] = phodymm_index_col # PhoDyMM's raw file row position (global, not per-chain)
        chunk_df = calculate_params(chunk_df) # Add all calculated parameters.
        processed_chunks.append(chunk_df)
    final_df = pd.concat(processed_chunks)
    final_df = add_interior_mass_and_positions(final_df) # Needs every planet of the system present at once.

    with pd.option_context('mode.chained_assignment', None):
        final_system_df = system_params(final_df) # Add system-wide value and comparison columns.
        final_system_df = rowe_table_attach(koi,final_system_df) # Add table from Lissauer et al.
        final_system_df = is_in_hsu(final_system_df)
        final_system_df = find_hidden_planet(koi,final_system_df) # Flag which planet (if any) is a "hidden" planet.
        final_system_df = final_system_df.drop("Unnamed: 0", axis=1) # Get rid of read-in column.
        id_number_identifier = final_system_df["chisq_rank"].astype(int)
        # Build the "KMDC index" as KOI-integer-part (XXXX, zero-padded) + KOI's
        # planet suffix (YY) + chisq_rank (Z, zero-padded) directly from KOI's own
        # "." split -- NOT by concatenating KOI+rank into one string and trying to
        # split that back apart, which doesn't work: there's no separator between
        # the KOI portion and the rank digits once they're joined, so a later
        # split("_") always comes up empty (str[1] on a 1-element list is NaN, and
        # .str.zfill on NaN raises).
        # .reindex(columns=[0, 1]) guards against the case where NOT A SINGLE row in
        # this system matched Rowe's table (KOI all-NaN) -- str.split(...,expand=True)
        # then produces only column 0, since there's no "." anywhere to split on, and
        # koi_parts[1] below would raise KeyError. Reindexing adds column 1 back as
        # NaN in that case, which flows into real_kmdc_index as NaN for every row --
        # the correct outcome, since none of those rows have a real KOI-based index.
        koi_parts = final_system_df["KOI"].astype(str).str.split(".", n=1, expand=True).reindex(columns=[0, 1])
        real_kmdc_index = (
            koi_parts[0].str.zfill(4)                          # XXXX padded
            + koi_parts[1]                                     # YY
            + id_number_identifier.astype(str).str.zfill(4)    # Z padded
        )

        # A hidden planet (no Rowe-table match) has no real KOI, so "KOI" is
        # NaN for it and real_kmdc_index above is NaN too. Give it a
        # substitute in the same format as real entries instead -- the
        # system's own koi number (known regardless of whether any given
        # planet in it matched Rowe's table), a placeholder ".0" decimal
        # (zero-padded to "00" to match the same 2-digit suffix width real
        # entries use), and chisq_rank.
        hidden_mask = final_system_df["is_hidden_planet"] == 1
        hidden_kmdc_index = (
            koi.zfill(4)
            + "00"
            + id_number_identifier.astype(str).str.zfill(4)
        )
        final_system_df["kmdc_index"] = real_kmdc_index.where(~hidden_mask, hidden_kmdc_index)
        # phodymm_index is already correctly populated at this point -- it was
        # captured in make_df_from_subsample directly from raw_data_df's own
        # row position in the raw file, and carried through unchanged ever
        # since (see the comment there). It is NOT the same thing as
        # step_number (which resets/repeats per chain) and is deliberately
        # left alone here rather than derived from final_system_df.index,
        # which is just this pipeline's own internal bookkeeping position by
        # this point (scrambled by the merge/groupby/concat calls earlier in
        # the pipeline) and has no relationship to PhoDyMM's own row order.
        final_system_df = final_system_df.reset_index(drop=True) # remove the internal bookkeeping index and set to default range
        final_system_df = final_system_df[col_headers] # Rearrange columns to be more legible
        return final_system_df


def column_rename(df):
    """Rename the columns of the df to be more legible."""
    num_columns = len(df.columns)
    for n, column in enumerate(df.columns):
        if num_columns-7 <= n:
            if n%9 == 0:
#                 print(column)
                assert(column=='M$_s$')
                df.rename(columns={column: 'M_s'}, inplace=True) # star's mass (in solar masses)
            elif n%9 == 1:
                df.rename(columns={column: 'R_s'}, inplace=True) # star's radius (in solar radii)
            elif n%9 == 2:
                df.rename(columns={column: 'c_1'}, inplace=True) # limb darkening 1
            elif n%9 == 3:
                df.rename(columns={column: 'c_2'}, inplace=True) # limb darkening 2
            else:
                break

        if n%9 == 0:
            df.rename(columns={column: f'planet'}, inplace=True) # indicates which member of the system it is
        elif n%9 == 1:
            df.rename(columns={column: f'Period_days'}, inplace=True) # planet's period (in days)
        elif n%9 == 2:
            df.rename(columns={column: f'T_0'}, inplace=True) # time of transit 0 (days in the BKJD system)
        elif n%9 == 3:
            df.rename(columns={column: f'sqrt(e)_cos(omega)'}, inplace=True)
        elif n%9 == 4:
            df.rename(columns={column: f'sqrt(e)_sin(omega)'}, inplace=True)
        elif n%9 == 5:
            df.rename(columns={column: f'i'}, inplace=True) # inclination (degrees)
        elif n%9 == 6:
            df.rename(columns={column: f'Omega'}, inplace=True) # longitudiny of the mean node (default 0 degrees)
        elif n%9 == 7:
            df.rename(columns={column: f'M_pJ'}, inplace=True) # planet's mass (in Jupiter masses)
        elif n%9 == 8:
            df.rename(columns={column: f'R_p/R_s'}, inplace=True) # planet's radius / star's radius
        else:
            raise ValueError("how did we get here?")

    return df


def calculate_params(df):
    """Calculate various planetary parameters and append to the df for convenient analysis."""
  ## masses, radii , and densities
    df['R_pJ'] = df['R_s'] * df['R_p/R_s'] * (1/RJTORS) # radius of planet (Jupiter radii)
    df['R_pE'] = df['R_pJ']*RJTORE # radius of planet (Earth radii)
    df['M_pE'] = df['M_pJ'] / METOMJ # mass of planet (Earth masses)
    df['rho_p'] = df['M_pE'] * MEG / ((4/3) * np.pi * (df['R_pE']*RECM)**3) # planetary density (g/cm^3)
    df['rho_s'] = (df['M_s']*MSKG*1000) / ((4/3) * np.pi * (df['R_s']*RSCM)**3) # stellar density (g/cm^3)
    df['M_p/M_s'] = df['M_pE'] / (df['M_s']/METOMS) # mass of planet / mass of star

 ## orbital angles
    df['Omega'] = df['Omega'] % 360
    df['e'] = df['sqrt(e)_cos(omega)']**2 + df['sqrt(e)_sin(omega)']**2 # eccentricity
    df['omega'] = (np.arctan2(df['sqrt(e)_sin(omega)'], df['sqrt(e)_cos(omega)']) * 180/np.pi) % 360 # argument of periapse (degrees)
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
    df['b_trans'] = (df['a_R_s'] * np.cos(df['i']*np.pi/180)) * ((1-df['e']**2)/(1+df['e']*np.sin(df['omega']*np.pi/180)))  # transit impact parameter
    df['b_occ'] = (df['a_R_s'] * np.cos(df['i']*np.pi/180)) * ((1-df['e']**2)/(1-df['e']*np.sin(df['omega']*np.pi/180))) # occultation impact parameter
    df['p_trans'] = ((df['R_s'] * RSAU + df['R_pJ']*RJAU) / df['a_AU']) * ((1+df['e']*np.sin(df['omega']*np.pi/180)) / (1-df['e']**2)) # transit probability
    df['p_occ'] = ((df['R_s'] * RSAU + df['R_pJ']*RJAU) / df['a_AU']) * ((1-df['e']*np.sin(df['omega']*np.pi/180)) / (1-df['e']**2)) # occultation probability
    df['T_total_hr'] = 24 * (df['Period_days'] / np.pi) * np.arcsin((df['R_s']*RSAU/df['a_AU'])*(np.sqrt((1+ df['R_p/R_s'])**2 - df['b_trans']**2)/np.sin(df['i']*np.pi/180))) * ((np.sqrt(1-df['e']**2))/(1+df['e']*np.sin(df['omega']*np.pi/180))) # total duration of transit (t4 - t1)
    df['T_full_hr'] = 24 * (df['Period_days'] / np.pi) * np.arcsin((df['R_s']*RSAU/df['a_AU'])*(np.sqrt(np.maximum(0,(1-df['R_p/R_s'])**2 - df['b_trans']**2))/np.sin(df['i']*np.pi/180))) * ((np.sqrt(1-df['e']**2))/(1+df['e']*np.sin(df['omega']*np.pi/180))) # full duration of transit (t3 - t2)
    df['K_RV'] = (2*np.pi*G/(df['Period_days']*24*60*60))**(1/3) * ((MSKG*df['M_pJ']*np.sin(df['i']*np.pi/180)/MSTOMJ)/((df['M_s']*MSKG)+(MSKG*df['M_pJ']/MSTOMJ))**(2/3)) * (1/(1-df['e']**2)**(1/2))  # amplitude of radial velocity variations    ## make sure units are right here. should be m/s

    df = occurrence_rate_params(df) # The Hsu et al occurrence rate parameters.

    df = mean_anomaly_corrections(df) # Per-row anomaly/eccentricity corrections (Hamann et al. formulation).

    # calculate_params + mean_anomaly_corrections just added ~60 columns one at
    # a time, which fragments the DataFrame's internal block layout (this is
    # what pandas' "DataFrame is highly fragmented" PerformanceWarning is
    # about -- it doesn't necessarily mean the LINE it's raised on is the
    # problem, just that whichever column got added next tripped the
    # fragmentation threshold). .copy() here consolidates it back into a
    # single block, which is exactly what the warning itself recommends, and
    # keeps every later single-column assignment in this pipeline from
    # continuing to compound on top of it.
    df = df.copy()

    return df


def occurrence_rate_params(df):
    """Attaches the occurrence rate parameters to a system df from Hsu et al 2018.

    occurrence_rates_hsu.csv is a rectangular, non-overlapping, exhaustive grid of
    radius bins x period bins (15 x 12 = 180 rows, covering 0 to 1e9 on both axes),
    so every row's (R_pE, Period_days) falls in exactly one cell. That means the bin
    lookup can be done with a single vectorized np.searchsorted per axis instead of a
    Python loop that re-scans all 180 ocdf rows for every row of df -- this is what
    made the original loop O(len(df) * len(ocdf)) instead of O(len(df) + len(ocdf)).
    """
    ocdf = _cached_read_csv("occurrence_rates_hsu.csv",engine='pyarrow')

    r_edges = np.sort(pd.unique(np.concatenate([ocdf['radius_lower'].values, ocdf['radius_upper'].values])))
    p_edges = np.sort(pd.unique(np.concatenate([ocdf['period_lower'].values, ocdf['period_upper'].values])))
    n_r, n_p = len(r_edges) - 1, len(p_edges) - 1

    # Build the (radius_bin, period_bin) -> value grids from ocdf.
    occ_grid = np.full((n_r, n_p), np.nan)
    Eor_grid = np.full((n_r, n_p), np.nan)
    eor_grid = np.full((n_r, n_p), np.nan)
    r_bin_of_row = np.searchsorted(r_edges, ocdf['radius_lower'].values, side='right') - 1
    p_bin_of_row = np.searchsorted(p_edges, ocdf['period_lower'].values, side='right') - 1
    occ_grid[r_bin_of_row, p_bin_of_row] = ocdf['occurrence'].values
    Eor_grid[r_bin_of_row, p_bin_of_row] = ocdf['+sigma'].values
    eor_grid[r_bin_of_row, p_bin_of_row] = ocdf['-sigma'].values

    # Look up every row of df in one shot. clip() guards against R_pE/Period_days
    # falling outside the grid's outer edges (shouldn't happen since the grid spans
    # 0 to 1e9, but avoids an out-of-bounds index if it ever does).
    r_bin = np.clip(np.searchsorted(r_edges, df['R_pE'].values, side='right') - 1, 0, n_r - 1)
    p_bin = np.clip(np.searchsorted(p_edges, df['Period_days'].values, side='right') - 1, 0, n_p - 1)

    df["occurrence_rate_hsu"] = occ_grid[r_bin, p_bin]
    df["E_or_hsu"] = Eor_grid[r_bin, p_bin]
    df["e_or_hsu"] = eor_grid[r_bin, p_bin]

    return df

def mean_anomaly_corrections(df):
    # turn omega into radians
    df['omega_rad'] = df['omega'] * np.pi / 180

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

    df['mean_anomaly_hamann_800'] = (df['mean_anomaly_hamann'] + df['mean_angular_motion'] * (800 - df['T_0'])) % (2*np.pi)
    df['mean_anomaly_hamann_850'] = (df['mean_anomaly_hamann'] + df['mean_angular_motion'] * (850 - df['T_0'])) % (2*np.pi)

    df['corrected_mean_anomaly_800'] = (df['false_mean_anomaly'] +   ((800 - df['T_0']) / df['Period_days'])* 2*np.pi ) % (2*np.pi)

    df['eccentric_anomaly_hamann_800'] = np.vectorize(
        eccentric_anomaly_from_mean
    )(
        df['e'].values,
        df['mean_anomaly_hamann_800'].values
    )

    df['eccentric_anomaly_hamann_850'] = np.vectorize(
        eccentric_anomaly_from_mean
    )(
        df['e'].values,
        df['mean_anomaly_hamann_850'].values
    )

    df['true_anomaly_hamann_800'] = np.vectorize(
        true_anomaly_from_eccentric
    )(
        df['e'].values,
        df['eccentric_anomaly_hamann_800'].values
    )

    df['true_anomaly_hamann_850'] = np.vectorize(
        true_anomaly_from_eccentric
    )(
        df['e'].values,
        df['eccentric_anomaly_hamann_850'].values
    )

    df['corrected_eccentric_anomaly_800'] = np.vectorize(
        eccentric_anomaly_from_mean
    )(
        df['e'].values,
        df['corrected_mean_anomaly_800'].values
    )

    df['corrected_true_anomaly_800'] = np.vectorize(
        true_anomaly_from_eccentric
    )(
        df['e'].values,
        df['corrected_eccentric_anomaly_800'].values
    )

    return df


def add_interior_mass_and_positions(df):
    """
    Computes each planet's interior-planet-mass-weighted mu, time of pericenter,
    and Jacobian cartesian state vectors.

    This has to run on the FULL system (all planets of a KOI already
    concatenated together), not per-planet -- the interior-mass cumulative sum
    needs every sibling planet's mass visible in the same dataframe. It groups
    by ('Chain#', 'step_number') rather than by star/KOI identifiers: every row
    passed into process_dataframe already belongs to the same system by
    construction (one koi is processed per call), and 'KIC'/'KOI' aren't
    populated yet at this point in the pipeline (they only exist after
    rowe_table_attach runs) -- and even after that point, a "hidden" planet
    with no Rowe-table match keeps KIC/KOI as NaN, which would silently exclude
    it from its own system's group and undercount interior mass for every
    other planet in that system.
    """
    df = df.copy()
    df['_orig_order'] = np.arange(len(df))  # remember row order; the sort below to get
                                             # ascending-period cumulative sums scrambles it
    df_sorted = df.sort_values(['Chain#', 'step_number', 'Period_days'])

    # Cumulative sum of planet mass within each (chain, step) group, ordered by period.
    # shift(1) so each planet gets the sum of planets *interior* to it, not including itself.
    df_sorted['interior_mass_pJ'] = (
        df_sorted.groupby(['Chain#', 'step_number'])['M_pJ']
        .transform(lambda x: x.cumsum().shift(1, fill_value=0))
    )

    # Now compute mu vectorially.
    # NOTE: must use GAU (Newton's constant in AU^3/day^2/Msun, PhoDyMM's own
    # convention -- see kg_constants.py), NOT the SI G. q/a_AU are in AU and
    # Tp/epoch are in days, and universal_cartesian's docstring is explicit
    # that mu must be in au^3/day^2 to match. Using SI G here (as originally
    # written) with masses left in solar masses is dimensionally wrong -- it
    # silently produced a mu about 8 orders of magnitude too small, which
    # would have produced a nonsense Tp and garbage x/y/z/vx/vy/vz for every
    # planet (verified: it turns a real ~4 day period into an apparent
    # ~8600 day period under Kepler's third law).
    df_sorted['mu'] = (
        GAU * (
            df_sorted['M_s']
        + df_sorted['M_pJ']    / MSTOMJ
        + df_sorted['interior_mass_pJ'] / MSTOMJ
        )
    )

    df_sorted['q'] = df_sorted['a_AU'] * (1 - df_sorted['e'])

    df_sorted['Tp'] = 800 - df_sorted['corrected_mean_anomaly_800'] * ((df_sorted['a_AU'])**3 / df_sorted['mu'] )**0.5


    # print("mu == 0:", (df_sorted['mu'] == 0).sum())
    # print("mu NaN:", df_sorted['mu'].isna().sum())
    # print("q == 0:", (df_sorted['q'] == 0).sum())
    # print("q NaN:", df_sorted['q'].isna().sum())
    # print("e == 0:", (df_sorted['e'] == 0).sum())

    positions = np.vectorize(universal_cartesian,otypes=[float,float,float,float,float,float])

    x, y, z, vx, vy, vz = positions(
        df_sorted['mu'].values,
        df_sorted['q'].values,
        df_sorted['e'].values,
        df_sorted['i'].values*np.pi/180,
        df_sorted['Omega'].values*np.pi/180,
        df_sorted['omega'].values*np.pi/180,
        df_sorted['Tp'].values,
        800
    )

    ### for some reason, universal_cartesian gives the NEGATIVE of the correct cartesion jacobean coords, so that's weird

    df_sorted['x']  = -x
    df_sorted['y']  = -y
    df_sorted['z']  = -z
    df_sorted['vx'] = -vx
    df_sorted['vy'] = -vy
    df_sorted['vz'] = -vz

    # restore original row order (safe even with duplicate index labels, unlike sort_index/.loc)
    df = df_sorted.sort_values('_orig_order').drop(columns=['_orig_order'])
    return df


def rowe_table_attach(koi,df):
    """Attaches Jason Rowe's table to a system df (from Lissauer et al 2024)."""
    rowe_df = _cached_read_csv("rowe_table_final.csv", low_memory=False)

    # Add Jason Rowe's columns as one pre-named block of NaNs, instead of the
    # old add-num_new_cols-placeholder-columns-then-rename-them-all-by-position
    # trick. That two-step version is what pandas' "DataFrame is highly
    # fragmented" PerformanceWarning was pointing at: `df[new_cols] = ...`
    # inserted ~85 columns one block at a time onto a df that had already
    # picked up dozens of individual columns from calculate_params, and the
    # positional rename afterward depended on column order lining up exactly
    # right. A single concat with the real column names already attached is
    # both faster (one consolidated block instead of insert-then-relabel) and
    # safer (no positional-rename step that could silently mislabel columns).
    # Use None (not np.nan) so these start out as object dtype, same as the
    # original placeholder trick -- several Rowe columns hold strings
    # (Kepler name, Source_rowe, Status_rowe, ...), and a float64-typed
    # column can't later be assigned a string value.
    rowe_cols_df = pd.DataFrame(
        [[None] * len(rowe_df.columns)], index=df.index, columns=rowe_df.columns
    )
    df = pd.concat([df, rowe_cols_df], axis=1)
    row_rowe_match = rowe_df.loc[(float(koi) == round(rowe_df["KOI"]))]

    # Attach the table values to each different planet. A planet with no match in
    # Jason Rowe's table (e.g. a "hidden" planet used only to fit TTVs, with no
    # corresponding Kepler candidate) is kept in df -- its Rowe columns are just
    # left blank. See find_hidden_planet() for how hidden planets are actually
    # identified and flagged.
    unmatched_planets = []
    for planet in df["planet"].unique():

        mask = (
            (row_rowe_match["Period_days_rowe"] > np.mean(df[df["planet"] == planet]["Period_days"]) - 0.2) &
            (row_rowe_match["Period_days_rowe"] < np.mean(df[df["planet"] == planet]["Period_days"]) + 0.2)
        )

        if not mask.any():
            unmatched_planets.append(planet)
            continue

        df.loc[df["planet"] == planet, "KIC":"e_BZ*_rowe"] = row_rowe_match.loc[mask].loc[:,"KIC":"e_BZ*_rowe"].values

    # Fallback for planets the +/-0.2 day period-proximity mask above couldn't catch:
    # single-transit ("mono-transiting") candidates, whose orbital period is
    # essentially unconstrained by the data. Jason Rowe's table marks these rows
    # with a NEGATIVE Period_days_rowe (a sentinel, not a real measured period)
    # instead of leaving the period blank -- so a fitted "long-distance" planet's
    # period estimate can be arbitrarily far from that sentinel, which is exactly
    # why the period-proximity mask misses them. We can't match these by period at
    # all, so instead: if this koi has exactly one still-unmatched planet left over
    # AND exactly one not-yet-used mono-transiting Rowe row, pair them up directly.
    # If there's more than one of either, there's no period information left to
    # disambiguate which unmatched planet goes with which mono-transit candidate,
    # so we deliberately leave all of them blank rather than guess wrong.
    if unmatched_planets:
        used_mask = pd.Series(False, index=row_rowe_match.index)
        for planet in df["planet"].unique():
            if planet in unmatched_planets:
                continue
            m = (
                (row_rowe_match["Period_days_rowe"] > np.mean(df[df["planet"] == planet]["Period_days"]) - 0.2) &
                (row_rowe_match["Period_days_rowe"] < np.mean(df[df["planet"] == planet]["Period_days"]) + 0.2)
            )
            used_mask = used_mask | m
        mono_candidates = row_rowe_match.loc[(row_rowe_match["Period_days_rowe"] < 0) & (~used_mask)]

        if len(unmatched_planets) == 1 and len(mono_candidates) == 1:
            planet = unmatched_planets[0]
            df.loc[df["planet"] == planet, "KIC":"e_BZ*_rowe"] = mono_candidates.loc[:, "KIC":"e_BZ*_rowe"].values
            unmatched_planets = []
        elif len(mono_candidates) > 0:
            print(
                f"koi {koi}: {len(unmatched_planets)} planet(s) unmatched by period and "
                f"{len(mono_candidates)} mono-transiting Rowe candidate(s) available -- "
                f"ambiguous with no period to disambiguate by, leaving all unmatched."
            )

    for planet in unmatched_planets:
        print(f"No Rowe table match found for koi {koi} planet {planet}; leaving its Rowe columns blank.")

    # Flag which planets are attached to a mono-transiting (single-transit, period
    # unconstrained) Rowe candidate, i.e. whichever row ended up with a negative
    # Period_days_rowe -- whether it got there through the normal period-proximity
    # match above (shouldn't normally happen, since a real fitted period is always
    # positive, but this is derived from the actual attached value rather than
    # "did we take the fallback branch" so it stays correct either way) or through
    # the mono-transit fallback just above. Unmatched planets (Period_days_rowe
    # still blank/None) are NOT flagged -- we don't know their status at all.
    df["is_monotransiting"] = (pd.to_numeric(df["Period_days_rowe"], errors="coerce") < 0).astype(int)

    return df


def is_in_hsu(df):
    """Creates a column which marks whether a system is in the Hsu et al. catalog or not."""
    hsu_star_df = _cached_read_csv("hsu_stellar_catalog_output.csv")
    df["hsu_flag"] = df["KIC"].isin(hsu_star_df['kepid']).astype(int)
    return df


def read_pldin_periods(path):
    """
    Parses a PhoDyMM .pldin file and returns a list of each planet's period (days).

    A .pldin file has a header row, then one row per planet (planet id,
    period, T0, e, i, Omega, omega, mp, ...), then stellar-parameter rows
    ("<value> ; <description>", e.g. "0.821... ; Mstar (M_sol)") and a final
    "; Comments: ..." row.

    This used to assume planet rows were prefixed with "> " (`if not
    line.startswith(">")`), which is wrong for real PhoDyMM output: a real
    koi0481_noHidden.pldin has the header and planet rows each *ending* in a
    bare ">" instead, e.g.
        "0.1     4.921362790...     791.368709...     ...     >"
    Since no line there ever starts with ">", the old check silently threw
    away every row and returned an empty list -- and because `any(...)` over
    an empty list is always False, every single planet in the system got
    misflagged as hidden (not just the real one), instead of just leaving
    no_hidden_periods correctly populated. This strips ">" from wherever it
    appears (leading, trailing, or absent) rather than assuming its position,
    and identifies a planet row by its first field parsing as a number (the
    header row's first field is the literal word "planet", which doesn't).
    """
    periods = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ";" in line:
                continue  # stellar-parameter or comment row, not a planet
            line = line.replace(">", "").strip()  # drop a leading OR trailing '>' column marker
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                float(fields[0])  # planet id -- also filters out the non-numeric header row
                period = float(fields[1])  # period (d) is the 2nd field
            except ValueError:
                continue
            periods.append(period)
    return periods


def find_hidden_planet(koi, df):
    """
    Flags which planet (if any) in a system is a "hidden" planet -- one PhoDyMM
    needed in the model to explain TTVs, but which has no corresponding Kepler
    candidate.

    PhoDyMM writes an alternate "noHidden" fit (one fewer planet) for any system
    where a hidden planet was used, at RAW_PATH/koi{koi}/koi{koi}_noHidden.pldin.
    If that file doesn't exist, the system has no hidden planet. If it does,
    whichever planet in the full fit has no period match among the noHidden
    fit's planets is the hidden one.
    """
    df["is_hidden_planet"] = 0

    no_hidden_path = os.path.join(RAW_PATH, "koi" + koi, f"koi{koi}_noHidden.pldin")
    if not os.path.exists(no_hidden_path):
        return df  # no noHidden fit was made -> this system has no hidden planet

    print(f"koi {koi}: this system has a hidden planet; checking which one it is...")

    no_hidden_periods = read_pldin_periods(no_hidden_path)
    period_tol = 0.6  # days -- tune per-KOI if it misclassifies

    hidden_planets = []
    for planet in df["planet"].unique():
        planet_period = np.mean(df.loc[df["planet"] == planet, "Period_days"])
        has_match = any(abs(planet_period - p) < period_tol for p in no_hidden_periods)
        if not has_match:
            hidden_planets.append(planet)
            df.loc[df["planet"] == planet, "is_hidden_planet"] = 1

    if len(hidden_planets) != 1:
        print(f"koi {koi}: expected exactly 1 hidden planet given koi{koi}_noHidden.pldin, found {len(hidden_planets)}: {hidden_planets}")

    return df


def system_params(df):
    """Calculates parameters related to each planet's ordering in its system and appends them to its df."""
    multiplicity = df['planet'].nunique()
    df['multiplicity'] = multiplicity  # number of planets per system

    # Preset all comparisons to -1 (which, unchanged, will flag that the comparison is invalid, i.e., innermost planet / its inner planet).
    df['P/Pin'] = -1.0
    df['P/Pout'] = -1.0
    df['Tdur/Tdurin'] = -1.0
    df['Tdur/Tdurout'] = -1.0
    df['R/Rin'] =  -1.0
    df['R/Rout'] = -1.0
    df['M/Min'] = -1.0
    df['M/Mout'] = -1.0
    df['rho/rhoin'] = -1.0
    df['rho/rhoout'] = -1.0
    df['i-iin'] = -1.0
    df['iout-i'] = -1.0
    df['xiin'] = -1.0
    df['xiout'] = -1.0
    df['distin_hillrad'] = -1.0
    df['distout_hillrad'] = -1.0
    df['distin_hillrad_e'] = -1.0
    df['distout_hillrad_e']= -1.0
    df['e/ein'] = -1.0
    df['eout/e'] = -1.0
    df['omega-omegain'] = -1.0
    df['omegaout-omega'] = -1.0

    # For each step of PhoDymm, there will be multiple planets; iterate through these.
    for i in df.index.unique():
        pair_df = df.loc[i]
        # For each system of multiple planets, we will need to make comparisons between them.
        for index, row in pair_df.iterrows():

            # Pre-find some values from df for convenience in calculations.
            planet=row["planet"]
            period=row["Period_days"]
            t_dur=row["T_total_hr"]
            radius=row["R_pE"]
            mass=row["M_pE"]
            density=row["rho_p"]
            inc=row["i"]
            sm_axis=row["a_AU"]
            omega=row["omega"]
            ecc=row["e"]
            planet_star_mass_ratio=row["M_p/M_s"]

            # Create ordering boolean checks.
            is_first = planet == 0.1
            is_last = planet*10 == multiplicity

            # Case: isn't the first planet
            if not is_first:

                df.loc[(df.index == i) & (df["planet"]==planet), "P/Pin"] = period / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "Period_days"].iloc[0]
                df.loc[(df.index == i) & (df["planet"]==planet), 'Tdur/Tdurin'] = t_dur / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "T_total_hr"].iloc[0]
                df.loc[(df.index == i) & (df["planet"]==planet), 'R/Rin'] = radius / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "R_pE"].iloc[0]
                df.loc[(df.index == i) & (df["planet"]==planet), 'M/Min'] = mass / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "M_pE"].iloc[0]
                df.loc[(df.index == i) & (df["planet"]==planet), 'rho/rhoin'] = density / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "rho_p"].iloc[0]
                df.loc[(df.index == i) & (df["planet"]==planet), 'i-iin'] = (inc - pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "i"].iloc[0]) % 360
                df.loc[(df.index == i) & (df["planet"]==planet), 'omega-omegain'] = (omega - pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "omega"].iloc[0]) % 360
                df.loc[(df.index == i) & (df["planet"]==planet), 'e/ein'] = ecc / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "e"].iloc[0]
                df.loc[(df.index == i) & (df["planet"]==planet), 'xiin'] = (pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "T_total_hr"].iloc[0] / t_dur) * (period / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "Period_days"].iloc[0])**(1/3)
                df.loc[(df.index == i) & (df["planet"]==planet), 'distin_hillrad'] = (sm_axis - pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"a_AU"].iloc[0]) / (((planet_star_mass_ratio+pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"M_p/M_s"].iloc[0])/3)**(1/3) * ((sm_axis + pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"a_AU"].iloc[0])/2))
                df.loc[(df.index == i) & (df["planet"]==planet), 'distin_hillrad_e'] = ((sm_axis * (1 - ecc)) - (pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"a_AU"].iloc[0] * (1 + pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"e"].iloc[0]))) / (((planet_star_mass_ratio+pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"M_p/M_s"].iloc[0])/3)**(1/3) * ((sm_axis + pair_df.loc[np.isclose(pair_df["planet"] , planet-0.1) ,"a_AU"].iloc[0])/2))

            # Case: isn't the last planet
            if not is_last:
                df.loc[(df.index == i) & (df["planet"]==planet), "P/Pout"] = period / pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "Period_days"].iloc[0]
                df.loc[(df.index == i) & (df["planet"]==planet), 'Tdur/Tdurout'] = t_dur / pair_df.loc[np.isclose(pair_df["planet"],planet+0.1), "T_total_hr"].iloc[0]
                df.loc[(df.index == i) & (df["planet"]==planet), 'R/Rout'] = radius / pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "R_pE"].iloc[0]
                df.loc[(df.index == i) & (df["planet"]==planet), 'M/Mout'] = mass / pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "M_pE"].iloc[0]
                df.loc[(df.index == i) & (df["planet"]==planet), 'rho/rhoout'] = density / pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "rho_p"].iloc[0]
                df.loc[(df.index == i) & (df["planet"]==planet), 'iout-i'] = (pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "i"].iloc[0] - inc) % 360
                df.loc[(df.index == i) & (df["planet"]==planet), 'omegaout-omega'] = (pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "omega"].iloc[0] - omega) % 360
                df.loc[(df.index == i) & (df["planet"]==planet), 'eout/e'] = pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "e"].iloc[0] / ecc
                df.loc[(df.index == i) & (df["planet"]==planet), 'xiout'] =  (t_dur / pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "T_total_hr"].iloc[0]) * (pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "Period_days"].iloc[0] / period)**(1/3)
                df.loc[(df.index == i) & (df["planet"]==planet), 'distout_hillrad'] = (pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"a_AU"].iloc[0] - sm_axis) / (((planet_star_mass_ratio+pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"M_p/M_s"].iloc[0])/3)**(1/3) * ((sm_axis + pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"a_AU"].iloc[0])/2))
                df.loc[(df.index == i) & (df["planet"]==planet), 'distout_hillrad_e'] = ((pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"a_AU"].iloc[0] * (1 - pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"e"].iloc[0])) - (sm_axis * (1 + ecc))) / (((planet_star_mass_ratio+pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"M_p/M_s"].iloc[0])/3)**(1/3) * ((sm_axis + pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"a_AU"].iloc[0])/2))

    # Another ~22 columns got declared one at a time above -- consolidate
    # again before returning (see the comment in calculate_params).
    df = df.copy()

    return df


def eccentric_anomaly_from_mean(e, M, tolerance=1e-14):
    """Convert mean anomaly to eccentric anomaly.

    Implemented from [A Practical Method for Solving the Kepler Equation][1]
    by Marc A. Murison from the U.S. Naval Observatory

    [1]: http://murison.alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf
    """
    MAX_ITERATIONS = 100
    Mnorm = fmod(M, 2 * np.pi)
    E0 = M + (-1 / 2 * e**3 + e + (e**2 + 3 / 2 * np.cos(M) * e**3) * np.cos(M)) * np.sin(M)
    dE = tolerance + 1
    count = 0
    while dE > tolerance:
        t1 = np.cos(E0)
        t2 = -1 + e * t1
        t3 = np.sin(E0)
        t4 = e * t3
        t5 = -E0 + t4 + Mnorm
        t6 = t5 / (1 / 2 * t5 * t4 / t2 + t2)
        E = E0 - t5 / ((1 / 2 * t3 - 1 / 6 * t1 * t6) * e * t6 + t2)
        dE = np.abs(E - E0)
        E0 = E
        count += 1
        if count == MAX_ITERATIONS:
            # ConvergenceError isn't defined/imported anywhere in this module -- raising it
            # would fail with NameError instead of surfacing the real non-convergence.
            raise RuntimeError(
                f"Did not converge after {MAX_ITERATIONS} iterations. (e={e!r}, M={M!r})"
            )
    return E


def true_anomaly_from_eccentric(e, E):
    """Convert eccentric anomaly to true anomaly."""
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))




def make_df_from_subsample(subsampled_rows,koi):
    """Creates an initial dataframe from a list of subsampled rows taken from a raw PhoDyMM output."""
    full_path = RAW_PATH + "koi" + koi + "/analysis_dir/dqa_allparam.csv"
    raw_data_df = pd.read_csv(full_path)

    # Capture PhoDyMM's own row position -- this file's natural 0..N-1 order,
    # the same thing "phodymm_index" has always been set from downstream --
    # as an explicit column before the merge below. A merge (like any join/
    # concat/groupby later in the pipeline) doesn't preserve raw_data_df's
    # original index, so if we don't save it here as real data it's gone for
    # good. This is deliberately NOT the same thing as "Unnamed: 0"/
    # step_number (the per-chain step count, which resets/repeats across
    # chains): phodymm_index is this row's position in the raw, whole-file
    # ordering, independent of which chain it came from.
    raw_data_df["phodymm_index"] = raw_data_df.index

    # This used to be:
    #   raw_data_df.loc[raw_data_df.index.isin(subsampled_rows["Unnamed: 0"])]
    # which is wrong. "Unnamed: 0" is the MCMC step number *within a chain*
    # (it repeats across chains -- e.g. koi 82's own subsamples file asks for
    # 1000 rows but only has 965 distinct "Unnamed: 0" values among them,
    # because the same step number legitimately shows up for many different
    # chains), while raw_data_df.index is a single globally-unique 0..N-1
    # row counter freshly assigned by this pd.read_csv call, with no
    # relationship to per-chain step numbering. Matching one against the
    # other worked only by numeric coincidence: for any step number shared
    # by two or more chains, `.isin()` could only return the one raw row
    # whose GLOBAL position happened to equal that number -- an arbitrary,
    # unrelated row, not the one kg_random_row_selector.py actually chose for
    # that chain. That silently corrupts Chain#, chisq, step_number, and
    # everything derived from them downstream (chisq_rank, phodymm_index,
    # kmdc_index) for every affected row.
    #
    # The real row identity is the (Chain#, "Unnamed: 0") PAIR -- both of
    # which kg_random_row_selector.py already wrote out precisely so this
    # join could be done correctly.
    key_cols = ["Chain#", "Unnamed: 0"]
    matched = raw_data_df.merge(subsampled_rows[key_cols], on=key_cols, how="inner")

    if len(matched) != len(subsampled_rows):
        print(
            f"koi {koi}: requested {len(subsampled_rows)} subsampled rows but "
            f"matched {len(matched)} rows in the raw data -- check for "
            f"duplicate (Chain#, step) keys or rows missing from the raw file."
        )

    return matched


def read_in_rows_write(breakpoints=False):
    """Tries to read in an output file from PhoDyMM; if anything errors for processing that file, this logs it."""
    for file in tqdm(os.listdir(SUBSAMPLED_PATH)):
        koi = find_koi(file)

        try:
            read_in_one_koi(koi)
            if breakpoints :
                input()

        except Exception as e:
            print("koi ",koi," failed")
            with open("subsampler_error_log.txt", "a") as file:
                file.write(koi+"\n")


def read_in_one_koi(koi):
    """Reads in a single PhoDymm output for a given KOI number."""
    print(f"starting koi: {koi}")
    subsampled_rows = pd.read_csv(SUBSAMPLED_PATH+koi+'_subsamples.csv')
    single_df_write(subsampled_rows,koi)
    print(f"finished koi: {koi}")


def single_df_write(subsampled_rows,koi):
    """"Creates a dataframe from one PhoDyMM output file, processes it, and then writes it to all_thin.csv ."""
    df = make_df_from_subsample(subsampled_rows,koi)
    df = process_dataframe(df,koi)
    write_header = not os.path.exists('thinned/KMDC.csv')
    df.to_csv('thinned/KMDC.csv', mode='a', header=write_header, index=False) # need to verify that index false works for the pipeline...


if __name__ == "__main__":
    if len(sys.argv) < 2:
        read_in_rows_write()
    elif len(sys.argv) == 2:
        if find_koi(sys.argv[1]) is not None:
            read_in_one_koi(find_koi(sys.argv[1]))
        else:
            print("enter a koi number to run the subsampler on!")
            sys.exit()
    elif len(sys.argv) > 2:
        read_in_rows_write(breakpoints=True) # With more than 2 arguments, the script is called in "debug mode" with breakpoints.
    else:
        print("invalid input")