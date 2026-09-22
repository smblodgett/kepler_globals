import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.csv as ar_csv
from mpi4py import MPI

from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent / "src"))
from kg_initialize_voxel_grid import process_singles_df, augment_stellar_df_with_fallbacks, repair_rowe_df_numeric_columns
from kg_constants import *

# All ranks read/filter the catalogs and call process_singles_df() together --
# it is a collective MPI operation (scatter/gather) that every rank must enter.
# Only rank 0 gets back a real DataFrame (every other rank gets None), so only
# rank 0 should go on to merge/derive columns/save -- see the rank==0 guard below.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


stellar_data_filename = "../data/berger_2020_keplerstellar.tsv"
rowe_stellar_data_filename ="../data/rowe_table_final.csv"
additional_stellar_data_filename = 'keplerstellar.csv'


def main():
    stellar_df = None
    singles_dr_df = None
    if rank == 0:
        stellar_df = pd.read_csv(stellar_data_filename,engine='pyarrow',delimiter='\t') # used to be from ../data/keplerstellar.csv, now is from Berger et al 2020


        print("len(stellar_df) before cuts: ",len(stellar_df))

        # Make the cuts to stellar catalog based off of temperature, logg

        # stellar_df = stellar_df[(stellar_df["Teff"]>4000) & (stellar_df["Teff"]<7000)]
        # stellar_df = stellar_df[(stellar_df["logg"]>4)]

        print("len(stellar_df) after hsu cuts: ",len(stellar_df))

        # NOTE: stellar_df here is ONLY Berger 2020 + keplerstellar.csv (additional stellar
        # info) -- it is NOT merged with Jason Rowe's table (rowe_table_final.csv). Rowe's
        # table has up to ~8 rows per KIC (one per candidate planet, not one per star), so
        # merging it onto stellar_df by KIC alone would silently duplicate stellar_df's rows
        # for every multi-candidate star. Rowe's table is instead read separately, below, as
        # the planet catalog itself (see the big comment right before `rowe_df = ...`).

        additional_stellar_df = pd.read_csv(additional_stellar_data_filename,engine='pyarrow')

        # keplerstellar.csv stacks five different stellar-catalog deliveries under one
        # 'kepid' column (q1_q17_dr25_stellar, q1_q17_dr24_stellar, q1_q16_stellar,
        # q1_q17_dr25_supp_stellar, q1_q12_stellar -- confirmed against this file: 990,244
        # rows for 200,038 distinct stars). Without filtering to one delivery, the merge
        # below fans every star's row out 5x. The file also has no 'KIC' column at all
        # (only 'kepid'), so the merge needs right_on='kepid', not 'KIC'.
        additional_stellar_df = additional_stellar_df[additional_stellar_df["st_delivname"]=="q1_q17_dr25_stellar"]

        stellar_df = stellar_df.merge(
                                    additional_stellar_df,
                                    left_on='KIC',
                                    right_on='kepid',
                                    how='left',
                                    # 'logg' exists (case-sensitive) in BOTH Berger's tsv and keplerstellar.csv --
                                    # without suffixes, that collision silently splits it into logg_x/logg_y and
                                    # neither survives as a plain 'logg' column, which kg_kmdc_col_headers.py's
                                    # final column list needs (it wants keplerstellar.csv's own 'logg', alongside
                                    # Berger's separately-used stellar params). Keep Berger's copy under a distinct
                                    # name (unused elsewhere) and let keplerstellar.csv's own 'logg' stay plain.
                                    suffixes=('_berger', '')
                                ).drop(columns=['kepid'])  # redundant with 'KIC' (same star) -- left in place, this
                                                            # collides with 'kepid' in every later kepid-keyed merge onto
                                                            # stellar_df, silently splitting THAT into kepid_x/kepid_y too.

        # ------------------------------------------------------------------------
        # Planet catalog: Jason Rowe's table (rowe_table_final.csv) is now the SOLE
        # source of planetary/transit-fit parameters for singles -- DR25
        # (q1_q17_dr25.csv) is no longer read at all. A data audit found that every
        # DR25 CONFIRMED/CANDIDATE KOI already has a row in Rowe's table, that none of
        # DR25's own stellar parameters (koi_steff/koi_slogg/koi_srad) are actually
        # used anywhere in this pipeline, and that Rowe's own stellar columns
        # (Teff_rowe/R*_rowe/M*_rowe/log(g)*_rowe/Z*_rowe) are essentially identical to
        # Berger et al. 2020's (median fractional difference ~0%, correlation >0.97) --
        # so nothing is lost by dropping DR25, and Rowe's table also recovers real
        # candidates DR25's snapshot excluded outright. We deliberately keep using
        # Berger's stellar density ('rho' in stellar_df, used inside
        # process_singles_df), NOT Rowe's rho*_rowe/rho*M_rowe -- those come from the
        # transit fit itself (a/R*), not an isochrone match, and empirically diverge a
        # lot from Berger's (correlation ~0.57 vs >0.97 for Teff/radius/mass/logg/feh).
        #
        # Per-KOI vetting: we deliberately do NOT filter out rows whose own current
        # disposition (first letter of Status_rowe) isn't 'P' (planet/candidate).
        # This catalog is meant to be a superset/successor to Rowe's own table, so
        # every KOI here -- including ones Rowe itself currently calls a false alarm
        # (F), too-low S/N (S), too-large radius (R), or too-massive-for-a-planet (M)
        # -- still gets run through the same posterior-sampling pipeline as everything
        # else, rather than being dropped up front. (If specific ones fail to converge
        # in the eccentricity/omega importance sampling inside process_singles_df,
        # that's worth its own try/except + a "failed to converge" flag -- not a
        # pre-filter here.)
        rowe_df = pd.read_csv(rowe_stellar_data_filename,engine='pyarrow') # Jason Rowe's expanded catalog: transit-fit + stellar parameters, one row per candidate.
        rowe_df = repair_rowe_df_numeric_columns(rowe_df)  # fix the rho*_rowe/E_Rp_rowe fixed-width-overflow rows

        # Systems PhoDyMM has already converged (handled entirely by kg_subsampler.py,
        # already in posterior-draw mode) shouldn't also be processed here.
        from create_nckmdc import find_converged_systems
        converged_multis_kois = find_converged_systems(verbose=False)
        rowe_df = rowe_df[~rowe_df['KIC'].isin(converged_multis_kois)]

        # Berger doesn't cover every KOI host star -- give every star Rowe lists a
        # usable stellar_df row (Rowe's own params for Source_rowe in {2,3}, DR25/
        # keplerstellar.csv for Source_rowe==0's "assume solar" placeholders) rather
        # than silently losing every planet around a star Berger's catalog missed.
        # See augment_stellar_df_with_fallbacks's own docstring for the full reasoning.
        stellar_df = augment_stellar_df_with_fallbacks(stellar_df, additional_stellar_df, rowe_df)

        rowe_df['multiplicity'] = rowe_df['KIC'].map(rowe_df['KIC'].value_counts())

        singles_dr_df = rowe_df[rowe_df['multiplicity']==1].copy()
        singles_dr_df['kepid'] = singles_dr_df['KIC']

        singles_dr_df['koi_prad'] = singles_dr_df['Rp_rowe']
        singles_dr_df['koi_prad_err1'] = singles_dr_df['e_Rp_rowe']
        singles_dr_df['koi_prad_err2'] = singles_dr_df['E_Rp_rowe']

        singles_dr_df['koi_period'] = singles_dr_df['Period_days_rowe']
        singles_dr_df['koi_period_err1'] = singles_dr_df['e_Period_rowe']
        singles_dr_df['koi_period_err2'] = singles_dr_df['e_Period_rowe']
        print("sum of empty error2s: ", np.sum(np.isnan(singles_dr_df['koi_period_err2']) ))

        singles_dr_df['koi_impact'] = singles_dr_df['b_rowe']
        singles_dr_df['koi_impact_err1'] = singles_dr_df['e_b_rowe']
        singles_dr_df['koi_impact_err2'] = singles_dr_df['E_b_rowe']

        singles_dr_df['koi_duration'] = singles_dr_df['TDur_rowe']
        singles_dr_df['koi_duration_err1'] = singles_dr_df['e_TDur_rowe']
        singles_dr_df['koi_duration_err2'] = singles_dr_df['e_TDur_rowe']


        print("len singles_dr_df) : ", len(singles_dr_df))
        print("sum singles_dr_df['kepid'].isin(stellar_df['KIC']) : ", np.sum(singles_dr_df["kepid"].isin(stellar_df['KIC'])))

        # augment_stellar_df_with_fallbacks (above) already gave every Source_rowe in
        # {0,2,3} KIC a stellar_df row, so this should now only drop a KIC in the rare
        # case Source_rowe==0 *and* keplerstellar.csv also has no row for it (logged
        # loudly by that function if it happens) -- not a silent "no Berger data" cut
        # anymore.
        print("len singles df before stellar_df-coverage filter: ", len(singles_dr_df))
        singles_dr_df = singles_dr_df[singles_dr_df["kepid"].isin(stellar_df['KIC'])]
        print("len singles df after stellar_df-coverage filter: ", len(singles_dr_df))



        # Remove the planets in the singles df that have nans in their period errors, since we need these for sampling the posteriors
        print("len singles df before removing period error: ", len(singles_dr_df))
        singles_dr_df = singles_dr_df[~(singles_dr_df["koi_period_err1"].isna() | singles_dr_df["koi_period_err2"].isna())]
        print("len singles df after removing period error: ", len(singles_dr_df))



        # Reset the index so we can iterate through singles df
        singles_dr_df = singles_dr_df.reset_index(drop=True)

        comm.bcast(stellar_df,root=0)
        comm.bcast(singles_dr_df,root=0)

        
    # Give the singles df the same cols as the multis df, sample ecc and omega for the singles
    processed_singles_dr_df = process_singles_df(singles_dr_df,stellar_df,0.01,10,seed=333,validation_graph=False,make_graphs=False)

    if rank == 0:
        print("finished processing!")

        df = processed_singles_dr_df.merge(
                                        stellar_df,
                                        left_on='kepid',
                                        right_on='KIC',
                                        how='left'
                                )

        # process_singles_df() only returns ["R_pE","Period_days","M_pE","e","omega","kepid"] --
        # it consumes koi_impact/koi_duration (and their error columns) internally but never
        # carries them through. Bring them back, along with KOI/Kepler and every *_rowe
        # column -- every row here already IS a Rowe row (singles_dr_df was built directly
        # from rowe_df above), so there's no separate attach/period-matching step needed
        # anymore, just carrying columns through on a plain merge on 'kepid' (singles_dr_df
        # has exactly one row per kepid, since it was already filtered to multiplicity == 1).
        _rowe_passthrough_cols = [c for c in rowe_df.columns if c not in ('KIC', 'Unnamed: 0', 'multiplicity')]
        df = df.merge(
                        singles_dr_df[['kepid', 'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
                                        'koi_duration', 'koi_duration_err1', 'koi_duration_err2']
                                        + _rowe_passthrough_cols],
                        on='kepid',
                        how='left'
                    )

        print("finished merging!")

        df['M_s'] = df['Mass']
        df['R_s'] = df['Rad']
        df['c_1'] = np.nan
        df['c_2'] = np.nan
        df['R_p/R_s'] = df['R_pE'] * RETORS / df['R_s']
        df['R_pJ'] = df['R_pE'] / RJTORE
        df['rho_p'] = df['M_pE'] * MEG / (4/3 * np.pi * (df['R_pE'] * RECM)**3)
        df['rho_s'] = 10**(df['rho']) * RHOS / 1000
        df['M_p/M_s'] = df['M_pE'] * MEKG / (df['M_s'] * MSKG)
        df['M_pJ'] = df['M_pE'] * METOMJ
        df['sqrt(e)_cos(omega)'] = np.sqrt(df['e']) * np.cos(df['omega'] * np.pi / 180)
        df['sqrt(e)_sin(omega)'] = np.sqrt(df['e']) * np.sin(df['omega'] * np.pi / 180)

        df['b_trans'] = np.random.normal(df['koi_impact'], np.max(np.abs([df['koi_impact_err1'], df['koi_impact_err2']]), axis=0), size=len(df))

        df['Omega'] = 0
        df['is_hidden_planet'] = 0
        df['is_monotransiting'] = (df['Period_days_rowe'] < 0).astype(int)
        df['planet'] = 0
        df['multiplicity'] = 1

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
        # Inverting kg_subsampler.py's forward relation
        # (b_trans = a_R_s * cos(i) * (1-e**2)/(1+e*sin(omega))) for cos(i), using the full
        # eccentricity/omega-corrected formula rather than the circular-orbit shortcut
        # (b_trans/a_R_s) -- this used to also carry a spurious extra R_s factor, fixed here.
        # b_trans is an independent normal draw (line above) and a_R_s is derived separately
        # from Period/M_s/M_pE/R_s, so cos(i) isn't guaranteed to land in [-1, 1] -- an unlucky
        # sample can push it just outside, which makes arccos silently return NaN (with a
        # RuntimeWarning) instead of raising. Clip into the valid domain, but log how many rows
        # needed it and by how much: a few hits at ~1e-10 are just floating-point noise, while
        # many rows or a large excess means b_trans and a_R_s are systematically inconsistent
        # for those planets and is worth investigating separately.
        cos_i = (df['b_trans'] * (1 + df['e'] * np.sin(df['omega'] * np.pi / 180))
                / (df['a_R_s'] * (1 - df['e']**2)))
        n_invalid = int((cos_i.abs() > 1).sum())
        if n_invalid:
            max_excess = float((cos_i.abs() - 1).clip(lower=0).max())
            print(f"[warn] {n_invalid}/{len(df)} rows have |cos(i)| > 1 "
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
        df["dilute"] = -1
        df["chisq"] = -1
        df["Chain#"] = np.nan
        df["chisq_rank"] = np.nan
        df["step_number"] = np.nan
        df["phodymm_index"] = np.nan
        df["phodymm_converged"] = np.nan

        ## add other cdpp columns and stuff from DR25 
        ## run a completeness rate calculation on all planets, add that column to all stuff



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



        id_number_identifier = df["M_pE"].rank(method='min', ascending=True) 
        koi_parts = df["KOI"].astype(str).str.split(".", n=1, expand=True).reindex(columns=[0, 1])
        real_kmdc_index = (
            koi_parts[0].str.zfill(4)                          # XXXX padded
            + koi_parts[1]                                     # YY
            + id_number_identifier.astype(str).str.zfill(4)    # Z padded
        )
        df['kmdc_index'] = real_kmdc_index


        from kg_kmdc_col_headers import col_headers
        df = df[col_headers]

        table = pa.Table.from_pandas(df)
        ar_csv.write_csv(table, f"thinned/KSDC.csv")

        print(f"Saved ksdc")

if __name__ == "__main__":
    main()