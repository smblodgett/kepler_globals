from mpi4py import MPI
import os

# rank initialization signature
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
print(f"[Rank {rank}/{size}] starting up")
print(os.system("hostname"))
# space out the walkers by a tenth of a second
import time
time.sleep(.02*rank) 


import os
import sys
import math
import numbers
import numpy as np
import pandas as pd
import json
from kg_utilities import ReadJson, mass_given_density_radius, repair_rowe_df_numeric_columns
from kg_griddefiner import RPMeoGrid, RPMeoVoxel
from kg_param_boundary_arrays import radius_grid_array, period_grid_array, mass_grid_array, eccentricity_grid_array, omega_grid_array
from kg_constants import G, RECM, RHOS, RSCM, MSKG, MEKG
from kg_plots import ecc_omega_singles_posterior_plot




class GridJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, numbers.Real):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj

        if isinstance(obj, pd.DataFrame):
            return obj.where(pd.notnull(obj), None).to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.where(pd.notnull(obj), None).to_dict()
        elif isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.number):
                return np.where(np.isnan(obj), None, obj).tolist()
            else:
                return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return {k: self.default(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.default(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self.default(i) for i in obj)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        return str(obj)


def _sigma_with_relative_fallback(central, err1, err2, relative_fraction=0.5):
    """
    Returns max(|err1|, |err2|), except where that comes out to exactly zero.
    Rowe's table reports BOTH error columns as literal 0.0 for a meaningful
    fraction of singles (2.25% for period, 7.2% for duration, 0.03% for radius --
    see the conversation this accompanies) -- not a genuinely perfectly-known
    measurement, just "uncertainty not computed" for KOIs whose fit didn't
    converge cleanly (mostly F/S-disposition KOIs). Drawing these as a degenerate
    fixed value would be false precision. Unlike b (whose central value can
    itself be an untrustworthy placeholder above the grazing limit -- see
    EccentricityOmegaConvergenceError), period/duration/radius central values are
    still usually real fit outputs even when their formal errors are missing, so
    keep the central value and substitute a wide but bounded relative uncertainty
    around it instead of discarding it: relative_fraction * |central| (default
    50%). This is a deliberate, conservative placeholder in the same spirit as
    the 100% relative density uncertainty already used for DR25-only stellar
    fallback rows in augment_stellar_df_with_fallbacks -- flagged the same way so
    it's easy to find and revisit later if real per-star errors turn up.
    """
    sigma = np.maximum(np.abs(err1), np.abs(err2))
    if sigma == 0:
        sigma = relative_fraction * np.abs(central)
    return sigma


class EccentricityOmegaConvergenceError(Exception):
    """
    Raised by sample_eccentricity_omega when NONE of its num_samples posterior draws
    produce a physically valid (inside > 0) transit geometry, so the importance-
    sampling weights are all exp(-inf) == 0 and can't be normalized (0/0 -> NaN,
    which used to blow up downstream in rng.choice with "Probabilities contain NaN").

    In practice this happens for KOIs whose reported impact parameter (b_rowe) sits
    above the grazing limit (b > 1 + Rp/Rstar) while Rowe's table also reports ZERO
    uncertainty on b (e_b_rowe == E_b_rowe == 0.0) -- a "no uncertainty computed"
    placeholder (639/9693 rows in rowe_table_final.csv), not a real, perfectly-known
    impact parameter. With zero scatter, every one of the num_samples draws of b is
    identical and > 1 + ratio, so inside > 0 is impossible for any of them regardless
    of the eccentricity/omega/period/duration draw. process_singles_df's per-planet
    loop catches this, logs the KIC, and skips that planet rather than crashing the
    whole (MPI-collective) run.
    """
    pass


def sample_eccentricity_omega(planet_star_radius_ratio, period, b, T_14,rho_star_true, rho_star_uncertainty,KIC_id,num_samples,rng,make_graphs=True):
    """
    Samples eccentricity and omega for a planet based on its radius and period, using the photoeccentric effect.
    
    See MacDougal, Gilbert, and Pettigura 2023. 
    """
    i=num_samples
    eccentricity = rng.uniform(0, 0.99,size=i)  # Sample eccentricity uniformly between 0 and 0.99
    omega = rng.uniform(0, 360,size=i)  # Sample omega uniformly between 0 and 360 degrees


    rho_star_sample = np.zeros(i)  # Initialize an array to store the sampled stellar densities

    inside = (((1+planet_star_radius_ratio)**2) - b**2)/ (np.sin(((T_14 / 24) * np.pi / period) * ( (1+ eccentricity*np.sin(omega * np.pi / 180))/ np.sqrt(1-eccentricity**2)))**2) + b**2
    
    valid = inside > 0

    rho_star_sample[valid] = (3 * np.pi / (G * (period[valid] * 24 * 3600)**2)) * (inside[valid])**1.5  # Sample stellar density based on the photoeccentric effect
    
    
    log_likelihood = np.full(i, -np.inf)  # Initialize log-likelihood array with negative infinity

    print("rho_star_sample: ",rho_star_sample)
    
    log_likelihood[valid] = -0.5 * (rho_star_sample[valid] - rho_star_true)**2 / rho_star_uncertainty**2  # Calculate log-likelihood based on the difference between the sampled and true stellar density

    print("log_likelihood: ",log_likelihood)

    unnormalized_weight = np.exp(log_likelihood)
    total_weight = np.sum(unnormalized_weight)

    print("weight :", unnormalized_weight / total_weight if total_weight > 0 else unnormalized_weight)

    if not np.isfinite(total_weight) or total_weight <= 0:
        raise EccentricityOmegaConvergenceError(
            f"KIC {KIC_id}: none of the {i} posterior draws produced a valid "
            f"(inside > 0) transit geometry -- importance-sampling weights are "
            f"all zero, can't be normalized. See EccentricityOmegaConvergenceError's "
            f"docstring for why this happens."
        )

    weight = unnormalized_weight / total_weight  # Normalize the weights

    indices = rng.choice(range(i), size=i, p=weight)  # Sample indices based on the weights

    eccentricity = eccentricity[indices]  # Sample eccentricity based on the weights
    omega = omega[indices]  # Sample omega based on the weights

    if make_graphs:
        ecc_omega_singles_posterior_plot(eccentricity,omega,KIC_id=KIC_id)

    ### how to get the eccentricity to be less elevated? How do we deweight the high eccentricities?
    ### cut out all samples with q < 2 stellar radii and reweight?

    return eccentricity, omega, rho_star_sample


def _sample_positive_normal(rng, loc, scale, size):
    """
    Draws from a Normal(loc, scale), resampling any non-positive values until
    the whole array comes out strictly positive.

    Used for radii (planet and stellar) in process_singles_df, which the rest
    of the pipeline assumes are strictly positive: mass_given_density_radius
    cubes the radius to get a mass, so a negative radius draw produces a
    negative mass outright, and downstream density/ratio calculations assume
    a positive stellar radius too. A plain rng.normal call has no floor at
    zero, so for any KOI whose reported radius uncertainty is large relative
    to its central value (common for small, faint singles) some fraction of
    the 1000 posterior draws would otherwise land at negative radius --
    silently corrupting that fraction of the planet's posterior with a
    negative mass rather than correctly representing its uncertainty.
    """
    if loc <= 0 and scale == 0:
        raise ValueError(
            f"_sample_positive_normal: loc={loc} <= 0 and scale=0 -- this can never "
            f"produce a positive draw (a zero-scale Normal is a point mass at loc, so "
            f"every 'resample' would just return loc again and loop forever). This "
            f"usually means the underlying catalog value is itself a 'no real "
            f"measurement' placeholder (e.g. Rp_rowe==0 with zero reported "
            f"uncertainty) that should be filtered out before reaching this function, "
            f"not something to draw a posterior from."
        )

    values = rng.normal(loc, scale, size=size)
    bad = values <= 0
    n_tries = 0
    while np.any(bad):
        n_tries += 1
        if n_tries > 10_000:
            raise ValueError(
                f"_sample_positive_normal: still {np.sum(bad)}/{size} non-positive "
                f"draws after {n_tries} resampling attempts (loc={loc}, scale={scale}) "
                f"-- giving up rather than looping forever."
            )
        values[bad] = rng.normal(loc, scale, size=np.sum(bad))
        bad = values <= 0
    return values




def augment_stellar_df_with_fallbacks(stellar_df, additional_stellar_df, rowe_df):
    """
    Berger et al. 2020 (stellar_df's primary source) only covers 186,301 of
    Kepler's ~200k target stars -- its Gaia-anchored pipeline doesn't reach
    every KOI host star. Left as-is, every planet around a star missing from
    Berger gets silently dropped upstream (process_singles_df looks stellar_df
    up by KIC and crashes -- empty-array IndexError -- on a star that isn't
    there), which throws away real Rowe-endorsed candidates along with the
    genuinely bad ones.

    This builds a Berger-shaped fallback row for every KIC in rowe_df that's
    missing from stellar_df, using rowe_table.txt's own documented Source_rowe
    flag (Note 13: 0=solar parameters, 1=DR25, 2=Berger et al. 2020, 3=Fulton &
    Petigura 2018) to pick the best real alternative:

      - Source_rowe == 2: Rowe's own R*_rowe/M*_rowe/Teff_rowe/log(g)*_rowe/
        rho*_rowe are real data here -- Rowe already carried a Berger value
        forward (usually for a KIC that isn't in *our* local Berger snapshot,
        a version/crossmatch difference). Use these directly.
      - Source_rowe == 0 or 3: despite Note 13 documenting 3 as "Fulton &
        Petigura (2018)", every single Source_rowe==3 row in this file --
        not just the ones missing from Berger -- has the identical values
        Teff=5780, R*=1.0, M*=1.0, log(g)*=4.5, rho*=0.0. That's not
        measured data with natural scatter, it's the same placeholder
        repeated, indistinguishable from what Source_rowe==0 ("solar
        parameters") actually is. Treat both the same way: fall back to
        keplerstellar.csv's DR25 stellar delivery (teff/radius/mass/dens) --
        real, if noisier-than-Gaia, independent measurements with full
        coverage for every one of these stars.

    Adds a 'stellar_source' column to every row (0=Berger, including
    Source_rowe==2's Rowe-carried-forward Berger values; 1=DR25/
    keplerstellar.csv fallback) so provenance stays traceable downstream.
    Any KIC that needs a fallback but isn't covered by keplerstellar.csv
    either (not currently expected, given the data on hand) is left out
    entirely and logged, rather than guessed at.
    """
    stellar_df = stellar_df.copy()
    stellar_df["stellar_source"] = 0  # every real Berger row

    RHOS_GCM3 = RHOS / 1000  # RHOS is solar density in kg/m^3; stellar_df's own
                             # 'rho' column (Berger's) and this fallback's 'rho'
                             # are both log10(density / solar density, g/cm^3).

    berger_kics = set(stellar_df["KIC"])
    rowe_unique = rowe_df.drop_duplicates("KIC").copy()
    missing = rowe_unique[~rowe_unique["KIC"].isin(berger_kics)].copy()

    if len(missing) == 0:
        return stellar_df

    fallback_frames = []

    # -- Source_rowe == 2: Rowe's own stellar columns are real data --
    rowe_ok = missing[missing["Source_rowe"] == 2].copy()
    if len(rowe_ok):
        # rowe_table_final.csv has a handful of columns (e.g. Kmag_rowe) with
        # occasional blank entries that make pandas infer 'object' dtype for
        # the whole column when read without engine='pyarrow' -- coerce the
        # numeric columns this function actually does arithmetic on, rather
        # than assume the caller read the file in a way that avoided it.
        num_cols = ["rho*_rowe", "E_rho*_rowe", "e_rho*_rowe", "R*_rowe",
                    "M*_rowe", "Teff_rowe", "log(g)*_rowe"]
        for c in num_cols:
            rowe_ok[c] = pd.to_numeric(rowe_ok[c], errors="coerce")
        rho_lin = rowe_ok["rho*_rowe"]
        E_lin = rowe_ok["E_rho*_rowe"]
        e_lin = rowe_ok["e_rho*_rowe"]
        fb = pd.DataFrame({
            "KIC": rowe_ok["KIC"].values,
            "Mass": rowe_ok["M*_rowe"].values,
            "Rad": rowe_ok["R*_rowe"].values,
            "Teff": rowe_ok["Teff_rowe"].values,
            "logg": rowe_ok["log(g)*_rowe"].values,
            "rho": np.log10(rho_lin / RHOS_GCM3).values,
            # E_rho*_rowe/e_rho*_rowe are genuine small linear (g/cm^3) deltas
            # around rho*_rowe (unlike Berger's own E_rho/e_rho, which
            # process_singles_df already treats as absolute upper/lower-bound
            # log-densities rather than deltas from rho -- match that same
            # convention here so these fallback rows are consumed identically
            # to a real Berger row).
            "E_rho": np.log10((rho_lin + E_lin) / RHOS_GCM3).values,
            "e_rho": np.log10((rho_lin - e_lin).clip(lower=1e-6) / RHOS_GCM3).values,
            "stellar_source": 0,
        })
        fallback_frames.append(fb)

    # -- Source_rowe in {0, 3} ("solar parameters", including the disguised-
    # as-Fulton-&-Petigura placeholder -- see docstring): fall back to
    # keplerstellar.csv's DR25 stellar delivery instead.
    dr25_ok = missing[missing["Source_rowe"].isin([0, 3])].copy()
    if len(dr25_ok):
        additional_stellar_df = additional_stellar_df.copy()
        for c in ["teff", "logg", "radius", "mass", "dens"]:
            additional_stellar_df[c] = pd.to_numeric(additional_stellar_df[c], errors="coerce")
        dr25_lookup = additional_stellar_df.drop_duplicates("kepid").set_index("kepid")
        # A KIC can be present in keplerstellar.csv but still have a NaN in
        # one of the fields we need (mass is the common one -- ~3k of its
        # 200,038 rows have no mass at all) -- .isin(index) alone wouldn't
        # catch that, and a NaN Mass/rho would silently propagate as "this
        # star's density is unknown" into process_singles_df. Require a
        # complete row, not just a matching KIC.
        complete_kics = dr25_lookup.dropna(subset=["teff", "logg", "radius", "mass", "dens"]).index
        have_dr25 = dr25_ok["KIC"].isin(complete_kics)
        if (~have_dr25).any():
            print(
                f"[warn] {(~have_dr25).sum()} Source_rowe in {{0,3}} KIC(s) have no "
                f"complete keplerstellar.csv row either (missing or absent "
                f"teff/logg/radius/mass/dens) -- no fallback available, still "
                f"dropped: {dr25_ok.loc[~have_dr25, 'KIC'].tolist()}"
            )
        dr25_ok = dr25_ok[have_dr25]
        if len(dr25_ok):
            matched = dr25_lookup.loc[dr25_ok["KIC"]]
            dens = matched["dens"].values
            rho_log = np.log10(dens / RHOS_GCM3)
            fb2 = pd.DataFrame({
                "KIC": dr25_ok["KIC"].values,
                "Mass": matched["mass"].values,
                "Rad": matched["radius"].values,
                "Teff": matched["teff"].values,
                "logg": matched["logg"].values,
                "rho": rho_log,
                # keplerstellar.csv carries no per-star density uncertainty at
                # all (point estimates only). Rather than fabricate false
                # precision, assume an order-unity (100%) relative
                # uncertainty -- i.e. the upper/lower bound sits the same
                # distance from solar density as the central value itself.
                # This is a deliberate, conservative placeholder, not a
                # measurement -- flagged via stellar_source==1 so it can be
                # revisited (e.g. if per-star DR25 errors turn up elsewhere).
                "E_rho": rho_log,
                "e_rho": rho_log,
                "stellar_source": 1,
            })
            fallback_frames.append(fb2)

    if not fallback_frames:
        return stellar_df

    fallback_df = pd.concat(fallback_frames, ignore_index=True)
    fallback_df = fallback_df.reindex(columns=stellar_df.columns)  # NaN for any
                                                                    # Berger-only
                                                                    # column we
                                                                    # have no
                                                                    # fallback for
                                                                    # (Dist/Age/...)

    n_before = len(stellar_df)
    stellar_df = pd.concat([stellar_df, fallback_df], ignore_index=True)
    print(
        f"augment_stellar_df_with_fallbacks: added {len(fallback_df)} fallback "
        f"stellar rows ({(fallback_df['stellar_source']==0).sum()} Berger-via-Rowe, "
        f"{(fallback_df['stellar_source']==1).sum()} DR25/keplerstellar.csv) -- "
        f"stellar_df grew from {n_before} to {len(stellar_df)} rows"
    )
    return stellar_df


def process_singles_df(singles_dr_df,stellar_df,lower_rho,upper_rho,seed=2222,validation_graph=True,make_graphs=True,comm=None):
    """
    comm: an MPI communicator (defaults to MPI.COMM_WORLD). Every rank in comm must
    call this function together -- the per-planet loop below (each planet needs its
    own million-draw eccentricity/omega posterior, which is the expensive part of
    voxel grid initialization) is scattered round-robin across all ranks and the
    results are gathered back onto rank 0. Only rank 0's return value is a DataFrame;
    every other rank gets None back.
    """

    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_sampling_draws = 1000000
    num_posteriors_per_planet = 1000

    n_planets = len(singles_dr_df)

    if validation_graph and rank == 0:
        ##### graphing GJ436 for validation - using Lanotte et al 2014
        validation_rng = np.random.default_rng(seed=seed)
        radius = validation_rng.normal(3.96,0.05,size=num_sampling_draws)
        period = validation_rng.normal(2.6438979,0.0000003,size=num_sampling_draws)
        b = validation_rng.normal(0.8521,0.0021,size=num_sampling_draws)
        T_14 = validation_rng.normal(0.04227*24,0.00016*24,size=num_sampling_draws)
        rho_star_true = (0.452 * MSKG * 1000) / ((4/3) * np.pi * (0.455 * RSCM)**3) * 1000
        rho_star_uncertainty_lower = ((0.452 - 0.012) * MSKG * 1000) / ((4/3) * np.pi * ((0.455 + 0.014) * RSCM)**3) * 1000
        rho_star_uncertainty_upper = ((0.452 + 0.014) * MSKG * 1000) / ((4/3) * np.pi * ((0.455 - 0.012) * RSCM)**3) * 1000
        rho_star_uncertainty = np.maximum(np.abs(rho_star_uncertainty_lower - rho_star_true), np.abs(rho_star_uncertainty_upper - rho_star_true))
        star_planet_radius_ratio = radius * RECM / (0.455 * RSCM)
        print("GJ rho star true: ",rho_star_true)
        print("GJ rho star uncertainty: ",rho_star_uncertainty)
        sample_eccentricity_omega(star_planet_radius_ratio, period, b, T_14,rho_star_true,rho_star_uncertainty,"GJ436",num_sampling_draws,validation_rng,make_graphs=make_graphs)
        #####

    # Distribute the per-planet loop across all ranks. Each planet gets its own
    # independent, reproducible RNG stream (spawned from a single SeedSequence keyed
    # on the planet's row index), so the draws are statistically independent no
    # matter how many ranks are used or which rank ends up processing which planet.
    # Planets are handed out round-robin -- the same scatter/gather pattern used in
    # RPMeoGrid.setup_completeness_grid.
    child_seeds = np.random.SeedSequence(seed).spawn(n_planets)

    tasks = list(range(n_planets))
    chunks = [tasks[r::size] for r in range(size)]
    my_chunk = comm.scatter(chunks, root=0)

    partial_rows = []

    n_failed_to_converge = 0

    for index in my_chunk:
        row = singles_dr_df.iloc[index]
        row_rng = np.random.default_rng(child_seeds[index])
        # Flag column: 1 when eccentricity/omega/inclination for this planet are
        # NaN placeholders rather than real sampled values, 0 otherwise. Set to 1
        # either when no real transit duration exists at all (koi_duration==0,
        # below) or when a real duration exists but eccentricity/omega importance
        # sampling still couldn't converge (EccentricityOmegaConvergenceError,
        # caught right at that call site below). Either way radius/period/mass/
        # stellar radius/stellar mass are still real sampled values -- only
        # e/omega/i are placeholders.
        ecc_omega_convergence_failed = 0

        # A handful of multi-planet-system KOIs (10/9693 total, all currently
        # flagged as false positives in Rowe's own disposition -- Status_rowe
        # starting with 'F' -- and absent from DR25 entirely, so there's no
        # fallback to recover a real value from) have Rp_rowe and BOTH its
        # reported uncertainties exactly 0.0: no real transit-fit radius was
        # ever measured, not a genuinely perfectly-known radius of zero.
        # _sample_positive_normal can't draw anything from a Normal(0, 0) --
        # it's a point mass at zero, and repeatedly resampling a non-positive
        # value would loop forever, so it raises instead of hanging. Radius
        # feeds into nearly everything else below (mass via
        # mass_given_density_radius, planet_star_radius_ratio, and from there
        # eccentricity/omega/inclination), so unlike the koi_duration==0 case
        # below, there's no way to keep most of the row real here -- only
        # period and stellar radius/mass are actually independent of planet
        # radius.
        radius_missing = (row["koi_prad"] == 0) and (row["koi_prad_err1"] == 0) and (row["koi_prad_err2"] == 0)
        if radius_missing:
            print(f"[rank {rank}] WARNING: kepid={row['kepid']} (singles index {index}) has "
                  f"koi_prad==0 with zero reported uncertainty on both sides and no DR25 "
                  f"fallback -- no real transit-fit radius is available. Leaving radius, "
                  f"mass, eccentricity, omega, and inclination as NaN for this planet; "
                  f"period and stellar radius/mass are still real sampled values.")
            radius = np.full(num_sampling_draws, np.nan)
        else:
            radius = _sample_positive_normal(row_rng, row["koi_prad"], _sigma_with_relative_fallback(row["koi_prad"], row["koi_prad_err1"], row["koi_prad_err2"]), num_sampling_draws)

        if row["koi_period"] < 0:
            period = row_rng.uniform(0.2,500,size=num_sampling_draws)
        else:
            period_sigma = _sigma_with_relative_fallback(row["koi_period"], row["koi_period_err1"], row["koi_period_err2"])
            period = row_rng.normal(row["koi_period"], period_sigma, size=num_sampling_draws)

        print(f"[rank {rank}] period with max abs error:", row["koi_period"], np.maximum(np.abs(row["koi_period_err1"]), np.abs(row["koi_period_err2"])))

        sigma_b = np.maximum(np.abs(row["koi_impact_err1"]), np.abs(row["koi_impact_err2"]))
        if sigma_b == 0:
            # Rowe's table reports e_b_rowe == E_b_rowe == 0.0 for 639/9693 rows
            # (mostly F/S-disposition KOIs) -- not a genuinely perfectly-known
            # impact parameter, just "uncertainty not computed". A literal zero
            # collapses b to one fixed value on every draw: harmless if that
            # value happens to be a normal, sub-grazing b, but for a few KOIs
            # that fixed value is itself above the grazing limit (b > 1+ratio),
            # which makes EVERY draw non-transiting and used to crash the whole
            # run (see EccentricityOmegaConvergenceError). Since b isn't actually
            # known for these, use an uninformative uniform prior over its
            # physically sensible range instead of trusting the placeholder.
            b = row_rng.uniform(0, 1, size=num_sampling_draws)
        else:
            b = row_rng.normal(row["koi_impact"], sigma_b, size=num_sampling_draws)
        T_14 = row_rng.normal(row["koi_duration"], _sigma_with_relative_fallback(row["koi_duration"], row["koi_duration_err1"], row["koi_duration_err2"]),size=num_sampling_draws)

        print(f"[rank {rank}] radius: ",radius)
        print(f"[rank {rank}] number of NaN in radius: ",np.sum(np.isnan(radius)))
        print(f"[rank {rank}] period: ",period)
        print(f"[rank {rank}] number of NaN in period: ",np.sum(np.isnan(period)))
        print(f"[rank {rank}] b: ",b)
        print(f"[rank {rank}] number of NaN in b: ",np.sum(np.isnan(b)))
        print(f"[rank {rank}] T_14: ",T_14)
        print(f"[rank {rank}] number of NaN in T_14: ",np.sum(np.isnan(T_14)))


        density = row_rng.uniform(lower_rho, upper_rho, size=num_sampling_draws)
        mass = mass_given_density_radius(density, radius)

        print(f"[rank {rank}] mass: ",mass)
        print(f"[rank {rank}] number of NaN in mass: ",np.sum(np.isnan(mass)))

        # make sure the units here are right, the log uncertainties are weird.

        ##### QUESTION for this process, for singles, should we be using the stellar density from the stellar_df, or should we be using the stellar density from the singles_dr_df? 
        # The singles_dr_df has a stellar density that is derived from the transit fit, while the stellar_df has a stellar density that is derived from the stellar parameters. 
        # I think we should be using the stellar_df, but I want to make sure.
        ##### 

        rho_star_true_log = stellar_df[stellar_df["KIC"]==row["kepid"]]["rho"].values[0] 
        rho_star_true = 10**(rho_star_true_log) * RHOS
        rho_star_upper_uncertainty = stellar_df[stellar_df["KIC"]==row["kepid"]]["E_rho"].values[0]
        rho_star_upper_uncertainty =  10**(rho_star_upper_uncertainty) * RHOS
        rho_star_lower_uncertainty = stellar_df[stellar_df["KIC"]==row["kepid"]]["e_rho"].values[0]
        rho_star_lower_uncertainty = - 10**(rho_star_lower_uncertainty) * RHOS
        rho_star_uncertainty = np.maximum(np.abs(rho_star_upper_uncertainty), np.abs(rho_star_lower_uncertainty))

        radius_star_val = stellar_df[stellar_df["KIC"]==row["kepid"]]["Rad"].values[0]
        radius_star_upper_uncertainty = stellar_df[stellar_df["KIC"]==row["kepid"]]["E_Rad"].values[0]
        radius_star_lower_uncertainty = stellar_df[stellar_df["KIC"]==row["kepid"]]["e_Rad"].values[0]
        radius_star_uncertainty = np.maximum(np.abs(radius_star_upper_uncertainty), np.abs(radius_star_lower_uncertainty))
        radius_star = _sample_positive_normal(row_rng, radius_star_val, radius_star_uncertainty, num_sampling_draws)


        planet_star_radius_ratio = radius * RECM / (radius_star * RSCM)

        print(f"[rank {rank}] rho_star_true: ",rho_star_true)
        print(f"[rank {rank}] rho_star_uncertainty: ",rho_star_uncertainty)
        print(f"[rank {rank}] number of NaN in rho_star_true: ",np.sum(np.isnan(rho_star_true)))
        print(f"[rank {rank}] number of NaN in rho_star_uncertainty: ",np.sum(np.isnan(rho_star_uncertainty)))

        duration_missing = (row["koi_duration"] == 0) and (row["koi_duration_err1"] == 0)
        if duration_missing or radius_missing:
            # Two distinct placeholder conditions both land here -- neither leaves a
            # trustworthy input to build an eccentricity/omega posterior from: no real
            # transit duration at all (duration_missing -- koi_duration==0 with its own
            # uncertainty also 0; would otherwise feed straight into a sin(0)==0
            # division-by-zero in this function's "inside" formula), or no real planet
            # radius at all (radius_missing, detected above -- planet_star_radius_ratio
            # is already NaN by this point, which would make every one of the
            # importance-sampling draws below invalid the same way a non-convergent
            # EccentricityOmegaConvergenceError would, just for an unrelated reason).
            # radius_missing already printed its own WARNING above; only print one here
            # for duration_missing, and skip straight to NaN either way instead of
            # calling sample_eccentricity_omega and routing this through the
            # convergence-failure flag below, which would mislabel the reason.
            if duration_missing:
                print(f"[rank {rank}] WARNING: kepid={row['kepid']} (singles index {index}) has "
                      f"koi_duration==0 with zero reported uncertainty and no DR25 fallback -- "
                      f"no real transit duration is available. Leaving eccentricity, omega, and "
                      f"inclination as NaN for this planet.")
            eccentricity = np.full(num_sampling_draws, np.nan)
            omega = np.full(num_sampling_draws, np.nan)
        else:
            try:
                eccentricity, omega, rho_star_sample = sample_eccentricity_omega(planet_star_radius_ratio, period, b, T_14,rho_star_true,rho_star_uncertainty,row["kepid"],num_sampling_draws,row_rng,make_graphs=make_graphs)
            except EccentricityOmegaConvergenceError as e:
                # A handful of KOIs have a reported impact parameter above the
                # grazing limit with zero reported uncertainty on b (see that
                # exception's docstring) -- no valid eccentricity/omega posterior
                # can be built for them. Caught here, right at the call site (same
                # place as the koi_duration==0 case above), instead of further out,
                # so this one planet's already-sampled radius/period/mass/stellar
                # values are kept rather than the whole row being discarded (which
                # is what catching this around the entire per-planet block used to do).
                n_failed_to_converge += 1
                ecc_omega_convergence_failed = 1
                print(f"[rank {rank}] WARNING: kepid={row['kepid']} (singles index {index}) "
                      f"-- eccentricity/omega sampling did not converge: {e}. Leaving "
                      f"eccentricity, omega, and inclination as NaN for this planet; radius, "
                      f"period, mass, and stellar radius/mass are still real sampled values.")
                eccentricity = np.full(num_sampling_draws, np.nan)
                omega = np.full(num_sampling_draws, np.nan)



        sampled_indices = row_rng.choice(range(num_sampling_draws), size=num_posteriors_per_planet, replace=True)

        i = np.arccos(np.clip(b * planet_star_radius_ratio * (1 + eccentricity * np.sin(omega * np.pi / 180)) / (1 - eccentricity**2), -1, 1)) * 180 / np.pi

        mass_star = stellar_df[stellar_df["KIC"]==row["kepid"]]["Mass"].values[0]
        mass_star_upper_uncertainty = stellar_df[stellar_df["KIC"]==row["kepid"]]["E_Mass"].values[0]
        mass_star_lower_uncertainty = stellar_df[stellar_df["KIC"]==row["kepid"]]["e_Mass"].values[0]
        mass_star_uncertainty = np.maximum(np.abs(mass_star_upper_uncertainty), np.abs(mass_star_lower_uncertainty))
        mass_star = _sample_positive_normal(row_rng, mass_star, mass_star_uncertainty, num_sampling_draws)

        radius = radius[sampled_indices]
        period = period[sampled_indices]
        mass = mass[sampled_indices]
        eccentricity = eccentricity[sampled_indices]
        omega = omega[sampled_indices]
        i = i[sampled_indices]
        radius_star = radius_star[sampled_indices]
        mass_star = mass_star[sampled_indices]

        row_result = np.array([radius, period, mass, eccentricity, omega, i,radius_star,mass_star,np.full(shape=num_posteriors_per_planet,fill_value=row["kepid"]),np.full(shape=num_posteriors_per_planet,fill_value=ecc_omega_convergence_failed)]).T
        partial_rows.append((index, row_result))
    if n_failed_to_converge:
        print(f"[rank {rank}] {n_failed_to_converge} planet(s) had eccentricity/omega sampling fail to converge -- kept with e/omega/i as NaN and ecc_omega_convergence_failed=1, not dropped")

    all_results = comm.gather(partial_rows, root=0)

    if rank == 0:
        flat = [item for sublist in all_results for item in sublist]
        # comm.gather preserves each rank's own order, but ranks only got every
        # size-th planet round-robin, so sort back into the original row order
        # before stitching the per-planet chunks into one array.
        flat.sort(key=lambda item: item[0])
        if n_planets == 0:
            final_singles_array = np.zeros((0,10))
        else:
            final_singles_array = np.concatenate([row_result for _, row_result in flat], axis=0)
        df = pd.DataFrame(final_singles_array, columns=["R_pE","Period_days","M_pE","e","omega","i","R_s","M_s","kepid","ecc_omega_convergence_failed"])
    else:
        df = None

    return df

def main(runprops):
    
    use_cache = os.path.isdir(runprops["voxel_data_folder"]) and not runprops["reload_KMDC"]

    # Define important variables for later in namespace
    voxel_grid = None
    stellar_df = None
    stellar_df_reduced = None
    final_kdc_df = None
    singles_dr_df = None
    comm = MPI.COMM_WORLD


    if comm.Get_rank() == 0:

        if not runprops["suppress_warnings"]: 
            if not use_cache:
                print("Warning! use_cache is",use_cache,"meaning that this run will take a long time!")
                print("Only run this way if your voxel data hasn't yet been cached.")
        
        # If the voxels don't have their data cached, then read in everything.
        if not use_cache:
            df = pd.read_csv(runprops["input_data_filename"],index_col=0,engine='pyarrow')
            if runprops["verbose"]: print("read in the catalog without caching (press enter to continue)")
            print("now we're caching it!")
            df = df[["R_pE","Period_days","M_pE","e","omega","KIC","rho_p","i","M_s","R_s","planet"]]#,"p_trans","MES_rowe"]]
            #df = create_probability_weighted(df)
            df.to_csv(runprops["input_data_folder"]+"/KMDC_RPMeo.csv")
            if runprops["verbose"]: print("data has been cached for future runs!")
            
        # Otherwise, you can just read in 1 voxel that has its data cached.    
        else:
            df = pd.read_csv(runprops["input_data_folder"]+"/KMDC_RPMeo.csv",index_col=0,engine='pyarrow')
            if runprops["verbose"]: print("read in cached df")

        print("full data df: ",df)

        # Get DR25 catalog for the inclusion of singles in the catalog.
        dr_df = pd.read_csv(runprops['dr_25_data_filename'],engine='pyarrow')

        # create a column for the multipliity of each DR25 planet
        dr_df['multiplicity'] = dr_df['kepid'].map(dr_df['kepid'].value_counts())


        # dr_df = dr_df[dr_df["koi_disposition"]=="CONFIRMED"] #??  - do we hit only the "confirmed" planets? I don't think so


        singles_dr_df = dr_df[dr_df["multiplicity"]==1]  #?? this is the cut for singles
        

        # Setup and load grid with data. If data is not cached, then cache data from whole grid into voxel dataframes.
        voxel_grid = RPMeoGrid(radius_grid_array, period_grid_array, mass_grid_array, eccentricity_grid_array, omega_grid_array)
        voxel_grid.setup_dataframes(df.columns)


        print("initialized voxel grid!")

        # Possible -- read in gaia data (though I don't think we need this right now)
        # gaia_df = pd.read_csv(runprops["gaia_data_filename"],delimiter='\t',header=1,engine='pyarrow')
        # gaia_df = gaia_df[["KIC","Mass","Teff","Rad"]]
        

        print("starting stellar df")
        # Read in the stellar df
        stellar_df = pd.read_csv(runprops["stellar_data_filename"],engine='pyarrow',delimiter='\t') # used to be from ../data/keplerstellar.csv, now is from Berger et al 2020

        # Read in the expanded stellar df, which has CDPP values
        additional_stellar_df = pd.read_csv(runprops["kepler_additional_stellar_filename"],engine='pyarrow') # where is this from.
        additional_stellar_df = additional_stellar_df[additional_stellar_df["st_delivname"]=="q1_q17_dr25_stellar"]

        print("len(stellar_df) before cuts: ",len(stellar_df))

        # Make the cuts to stellar catalog based off of temperature, logg
        stellar_df = stellar_df[(stellar_df["Teff"]>4000) & (stellar_df["Teff"]<7000)]
        stellar_df = stellar_df[(stellar_df["logg"]>4)]

        print("len(stellar_df) after hsu cuts: ",len(stellar_df))

        # Remove the stellar entries with nans in mass or temperature
        stellar_df = stellar_df[(~stellar_df["Mass"].isna())  & (~stellar_df["Teff"].isna())] # & (~stellar_df["limbdark_coeff1"].isna())]

        # Merge with the Rowe catalog to get the CDPP and dataspan values for each star
        stellar_df = stellar_df.merge(
                                    additional_stellar_df[
                                        ['kepid', 'dataspan', 'rrmscdpp01p5', 'rrmscdpp02p0', 'rrmscdpp02p5', 'rrmscdpp03p0',
                                        'rrmscdpp03p5', 'rrmscdpp04p5', 'rrmscdpp05p0', 'rrmscdpp06p0', 'rrmscdpp07p5',
                                        'rrmscdpp09p0', 'rrmscdpp10p5', 'rrmscdpp12p0', 'rrmscdpp12p5', 'rrmscdpp15p0']
                                    ],
                                    left_on='KIC',
                                    right_on='kepid',
                                    how='left'   # keeps all rows from stellar_df
                                )
        
        print("len(stellar_df) after removing nans: ",len(stellar_df))

        print("number of NaNs in rrms columns: ",stellar_df[['rrmscdpp01p5', 'rrmscdpp02p0', 'rrmscdpp02p5', 'rrmscdpp03p0',
                                        'rrmscdpp03p5', 'rrmscdpp04p5', 'rrmscdpp05p0', 'rrmscdpp06p0', 'rrmscdpp07p5',
                                        'rrmscdpp09p0', 'rrmscdpp10p5', 'rrmscdpp12p0', 'rrmscdpp12p5', 'rrmscdpp15p0']].isna().sum().sum())

        stellar_df = stellar_df.dropna(subset=['rrmscdpp01p5', 'rrmscdpp02p0', 'rrmscdpp02p5', 'rrmscdpp03p0',
                                        'rrmscdpp03p5', 'rrmscdpp04p5', 'rrmscdpp05p0', 'rrmscdpp06p0', 'rrmscdpp07p5',
                                        'rrmscdpp09p0', 'rrmscdpp10p5', 'rrmscdpp12p0', 'rrmscdpp12p5', 'rrmscdpp15p0'])

        print("len(stellar_df) after removing NaN in cdpp columns: ",len(stellar_df))

        

        print("number of infs in rrms columns: ",stellar_df[['rrmscdpp01p5', 'rrmscdpp02p0', 'rrmscdpp02p5', 'rrmscdpp03p0',
                                        'rrmscdpp03p5', 'rrmscdpp04p5', 'rrmscdpp05p0', 'rrmscdpp06p0', 'rrmscdpp07p5',
                                        'rrmscdpp09p0', 'rrmscdpp10p5', 'rrmscdpp12p0', 'rrmscdpp12p5', 'rrmscdpp15p0']].isin([np.inf, -np.inf]).sum().sum())

        
        print("stellar df cuts applied")

        # Get the KICs of the stars in the stellar df
        stellar_df_kics = stellar_df["KIC"]
        # Only include the planets in the multis df that are in the stellar df
        df = df[df["KIC"].isin(stellar_df_kics)]
        # Only include the planets in the singles that are in the stellar df
        singles_dr_df = singles_dr_df[singles_dr_df["kepid"].isin(stellar_df_kics)]

        print("len(singles_dr_df): ",len(singles_dr_df))
        print("nan in singles_dr_df periods: ",singles_dr_df["koi_period"].isna().sum())
        print("nan in singles_dr_df periods err1: ",singles_dr_df["koi_period_err1"].isna().sum())
        print("nan in singles_dr_df periods err2: ",singles_dr_df["koi_period_err2"].isna().sum())

        print("location of nan in errors: ",singles_dr_df[singles_dr_df["koi_period_err1"].isna() | singles_dr_df["koi_period_err2"].isna()]["kepid"])

        # Remove the planets in the singles df that have nans in their period errors, since we need these for sampling the posteriors
        singles_dr_df = singles_dr_df[~(singles_dr_df["koi_period_err1"].isna() | singles_dr_df["koi_period_err2"].isna())]
        # Reset the index so we can iterate through singles df
        singles_dr_df = singles_dr_df.reset_index(drop=True)

    # Broadcast what process_singles_df needs so every rank can take part in
    # distributing its per-planet eccentricity/omega sampling loop below -- that loop
    # is the expensive part of setup (a million draws per KOI), so it shouldn't run on
    # rank 0 alone.
    singles_dr_df = comm.bcast(singles_dr_df, root=0)
    stellar_df = comm.bcast(stellar_df, root=0)

    # Give the singles df the same cols as the multis df, sample ecc and omega for the
    # singles. All ranks call this together; process_singles_df scatters the planets
    # across ranks internally and gathers the result back onto rank 0.
    processed_singles_dr_df = process_singles_df(singles_dr_df,stellar_df,runprops["minimum_density"],runprops["maximum_density"],comm=comm)

    if comm.Get_rank() == 0:
        # Remove the planets with densities above or below a certain threshold, because they are unphysical


        print("length of df before requiring stability: ",len(df))
        if runprops["exclude_bad_densities"]:
            df = df[(df["rho_p"]<runprops["maximum_density"]) & (df["rho_p"]>runprops["minimum_density"])]
        print("length of df after excluding bad densities: ",len(df))


        print("length of df after before requiring stability: ",len(df))
        # Exclude any posterior draw that has a periastron less than 2 stellar radii
        sm_axis = (df["Period_days"] * 24 * 60 * 60)**(2/3) * (G / (4*np.pi**2))**(1/3) * (df["M_pE"] * MEKG + stellar_df.set_index("KIC").loc[df["KIC"],"Mass"].values * MSKG)**(1/3)
        periapsis = (1 - df["e"]) * sm_axis
        df = df[periapsis >= 2 * stellar_df.set_index("KIC").loc[df["KIC"],"Rad"].values]
        print("length of df after after requiring stability: ",len(df))

        print("length of singles df after before requiring stability: ",len(processed_singles_dr_df))
        sm_axis = (processed_singles_dr_df["Period_days"] * 24 * 60 * 60)**(2/3) * (G / (4*np.pi**2))**(1/3) * (processed_singles_dr_df["M_pE"] * MEKG + stellar_df.set_index("KIC").loc[processed_singles_dr_df["kepid"],"Mass"].values * MSKG)**(1/3)
        periapsis = (1 - processed_singles_dr_df["e"]) * sm_axis
        processed_singles_dr_df = processed_singles_dr_df[periapsis >= 2 * stellar_df.set_index("KIC").loc[processed_singles_dr_df["kepid"],"Rad"].values]
        print("length of processed_singles_dr_df after requiring stability: ",len(processed_singles_dr_df))


        # Define the "unique planet" column which is the combo of the KIC and planet number
        df['unique_planet'] = df['KIC'].astype(str) + "_" + df['planet'].astype(str)
        # Make a dict of how many unique planets there are, with the unique planet as the key and the count as the value
        kic_dict_multis = df['unique_planet'].value_counts().to_dict()

        print("kic_dict_multis: ", kic_dict_multis)

        # List of unique planets that have less than 50 samples, which we remove because they will be improperly weighted
        unique_planet_to_remove = [k for k, v in kic_dict_multis.items() if v < 50]

        # Actually remove these planets
        for unique_planet in unique_planet_to_remove:
            df = df[df['unique_planet'] != unique_planet]

        # Update the kic_dict_multis after removing the planets with less than 50 samples (necessary for reweighting procedure)
        kic_dict_multis = {k: v for k, v in kic_dict_multis.items() if v >= 50}

        print("kic_dict_multis: ", kic_dict_multis)
        
        # Give the singles a unique planet column as well
        processed_singles_dr_df['unique_planet'] = processed_singles_dr_df['kepid'].astype(int).astype(str) + "_" + "0.1"

        # Make a dict of how many unique planets are in singles
        kic_dict_singles = processed_singles_dr_df['unique_planet'].value_counts().to_dict()

        # List of unique planets that have less than 50 samples, which we remove because they will be improperly weighted
        unique_planet_to_remove = [k for k, v in kic_dict_singles.items() if v < 50]

        # Actually remove these planets
        for unique_planet in unique_planet_to_remove:
            processed_singles_dr_df = processed_singles_dr_df[processed_singles_dr_df['unique_planet'] != unique_planet]

        # Update the kic_dict_singles after removing the planets with less than 50 samples (necessary for reweighting procedure)
        kic_dict_singles= {k: v for k, v in kic_dict_singles.items() if v >= 50}

        print("kic_dict_singles: ", kic_dict_singles)

        # Merge the multis and singles count dicts
        kic_dict = kic_dict_multis | kic_dict_singles

        print("kic_dict: ", kic_dict)

        # Give the RPMeoGrid the kic_dict so that it can reweight the planets
        voxel_grid.set_kic_dict(kic_dict)

        # Concatenate the multis and processed singles dfs together to get the KDC - Kepler Dynamical Catalog
        final_kdc_df = pd.concat([df, processed_singles_dr_df.rename(columns={"kepid":"KIC"})], ignore_index=True)
        ######## ADD A FLAG TO SEE IF ITS A SINGLE OR A MULTI (FOR PLOTTING PURPOSES)

        # Where there doesn't exist a PhoDyMM value, fill with the best guess from the stellar catalog.
        final_kdc_df["M_s"] = final_kdc_df['M_s'].fillna(final_kdc_df["KIC"].map(stellar_df.set_index("KIC")["Mass"]))
        final_kdc_df["R_s"] = final_kdc_df['R_s'].fillna(final_kdc_df["KIC"].map(stellar_df.set_index("KIC")["Rad"]))

        final_kdc_df["Teff"] = final_kdc_df["KIC"].map(stellar_df.set_index("KIC")["Teff"])


        print("final_kdc_df: ",final_kdc_df)

        print("final_kdc_df columns: ",final_kdc_df.columns)
        
        print("length of df after matching to stellar catalog, filtering densities: ",len(final_kdc_df))

        # Add the data to the RPMeoGrid voxel grid object (this object will be written to a json, then read in for the model runs)
        voxel_grid.add_data(final_kdc_df)

        # Create a small stellar df with 1000 random stars, to set up the completeness grid. (could be expanded to entire stellar catalog)
        stellar_df_reduced=stellar_df.sample(n=250,random_state=44)


    voxel_grid = comm.bcast(voxel_grid,root=0)
    stellar_df_reduced = comm.bcast(stellar_df_reduced,root=0)

    print("broadcasted voxel grid and stellar df")
        
    voxel_grid.setup_completeness_grid(stellar_df_reduced,comm) # this is the kepler stellar catalog, which has the stellar radii and masses
    print("set up completeness grid")
    voxel_grid.setup_likelihood_grid()
    # MES_grid_plot(voxel_grid.p_detection_interp,voxel_grid.p_transit_interp,runprops["completeness_plot_folder"])
    
    if runprops["verbose"] and comm.Get_rank() == 0: print("MES grid has been set up!")

    comm.Barrier()

    if comm.Get_rank() == 0:
        grid_string = json.dumps(voxel_grid,cls=GridJSONEncoder)

        
        with open(runprops["voxel_json_filename"], "w") as f:
            f.write(grid_string)

        import pyarrow as pa
        import pyarrow.csv as csv

        stellar_table = pa.Table.from_pandas(stellar_df)
        csv.write_csv(stellar_table, "../data/keplerstellar_with_cuts.csv")

        # stellar_df.to_csv("../data/keplerstellar_with_cuts.csv")

        final_kdc_table = pa.Table.from_pandas(final_kdc_df)
        csv.write_csv(final_kdc_table, "../data/final_kdc.csv")

        # final_kdc_df.to_csv("../data/final_kdc.csv")

        final_kdc_df_columns = json.dumps(list(final_kdc_df.columns))
        with open('../data/dataframe_column_names.json', "w") as f:
            f.write(final_kdc_df_columns)
        
        print("Finished writing to json!")
    
    comm.Barrier()



if __name__ == "__main__":       
    
    # Verify the correct path script is being run from. 
    cwd = os.getcwd()
    print(cwd)        

    # Find the runprops file path. 
    if 'src' in cwd:
        runprops_filename = "../runs/param_runprops.txt"
    elif 'runs' in cwd:
        runprops_filename = "param_runprops.txt"
    elif 'results' in cwd:
        runprops_filename = "param_runprops.txt"
    else:
        print('you are not starting from a proper directory. you should run kg_run_param.py from a src, runs, or a results directory.')
        sys.exit(1)
    
    # Get runprops loaded in, find the initial guess file.
    getData = ReadJson(runprops_filename)
    runprops = getData.outProps()

    main(runprops)