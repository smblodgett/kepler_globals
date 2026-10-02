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
        dr25_full.csv's DR25 stellar delivery (teff/radius/mass/dens, plus
        dens_err1/dens_err2) first -- real, if noisier-than-Gaia,
        independent measurements, pulled straight from the archive rather
        than the old stripped-down local keplerstellar.csv extract. A
        remainder isn't in dr25_full.csv either (no complete
        teff/logg/radius/mass/dens/dens_err1/dens_err2 row there): rather
        than drop these real,
        Rowe-endorsed planet candidates outright, assign them the same
        solar reference values Note 13 already uses for this flag (Teff=
        5780, Mass=1.0, Rad=1.0, logg=4.5 -- density follows automatically,
        since solar density is exactly 0 in this function's own
        log10(rho/RHOS_GCM3) convention), with a deliberately wide
        uncertainty on density -- the only field this schema actually
        carries a propagated uncertainty for downstream -- taken from the
        real star-to-star scatter of Berger 2020's own ~186k-star
        population (computed at call time directly from stellar_df,
        before any fallback rows are appended), rather than reaching for a
        different external survey with its own separate selection
        function. These are stars that Rowe's own table, Berger 2020, AND
        DR25 all three separately failed to characterize, so there's a
        real chance they aren't ordinary FGK dwarfs at all -- this is a
        deliberately weak, wide prior for a genuinely unknown star, not a
        measurement.

    Adds a 'stellar_source' column to every row (0=Berger, including
    Source_rowe==2's Rowe-carried-forward Berger values; 1=DR25/
    dr25_full.csv fallback, with real per-star density uncertainties;
    2=solar reference values with Berger-population density scatter, for a
    KIC no real source covers) so provenance -- and which rows rest on an
    assumed prior rather than an actual measurement -- stays traceable
    downstream.
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
    # dr25_full.csv's DR25 stellar delivery instead.
    dr25_ok = missing[missing["Source_rowe"].isin([0, 3])].copy()
    if len(dr25_ok):
        additional_stellar_df = additional_stellar_df.copy()
        for c in ["teff", "logg", "radius", "mass", "dens", "dens_err1", "dens_err2"]:
            additional_stellar_df[c] = pd.to_numeric(additional_stellar_df[c], errors="coerce")
        dr25_lookup = additional_stellar_df.drop_duplicates("kepid").set_index("kepid")
        # A KIC can be present in dr25_full.csv but still have a NaN in one of
        # the fields we need (mass is the common one -- ~3k of its 200,038
        # rows have no mass at all; dens_err1/dens_err2 are separately absent
        # on ~244 rows that do have dens) -- .isin(index) alone wouldn't catch
        # that, and a NaN Mass/rho would silently propagate as "this star's
        # density is unknown" into process_singles_df. Require a complete
        # row, not just a matching KIC -- now including the per-star density
        # error columns, since those feed E_rho/e_rho directly below.
        complete_kics = dr25_lookup.dropna(subset=["teff", "logg", "radius", "mass", "dens", "dens_err1", "dens_err2"]).index
        have_dr25 = dr25_ok["KIC"].isin(complete_kics)
        still_missing = dr25_ok[~have_dr25].copy()
        if len(still_missing):
            # Neither Rowe's table, Berger 2020, nor DR25 has anything for
            # these KICs. Rather than drop them (their planet candidates
            # would otherwise vanish from the population inference
            # entirely -- a real selection effect, not a random loss),
            # assign the same solar reference values Note 13 already uses
            # for Source_rowe==0 ("solar parameters"), with a deliberately
            # wide density uncertainty taken from the real star-to-star
            # scatter of Berger 2020's own population (computed here, from
            # stellar_df itself, before any fallback rows are appended) --
            # not a different external survey's numbers, and not Berger's
            # own much tighter per-star fit precision. This is a weak,
            # honest prior for a genuinely unknown star, not a
            # measurement, and it's tagged stellar_source==2 (distinct
            # from stellar_source==1's real DR25 point estimates just
            # below) so it's trivial to exclude or test sensitivity to
            # downstream.
            print(
                f"[warn] {len(still_missing)} Source_rowe in {{0,3}} KIC(s) have no "
                f"complete dr25_full.csv row either (missing or absent "
                f"teff/logg/radius/mass/dens/dens_err1/dens_err2) -- assigned "
                f"solar reference values with Berger-population density "
                f"scatter instead of being dropped (stellar_source==2): "
                f"{still_missing['KIC'].tolist()}"
            )
            SOLAR_TEFF, SOLAR_MASS, SOLAR_RAD, SOLAR_LOGG = 5780.0, 1.0, 1.0, 4.5
            # Berger's own real population, already in this function's
            # log10(rho/RHOS_GCM3) convention -- solar itself is exactly 0
            # in that convention, so no unit conversion is needed here.
            rho_sd = stellar_df["rho"].std()
            n_fp = len(still_missing)
            fb3 = pd.DataFrame({
                "KIC": still_missing["KIC"].values,
                "Mass": np.full(n_fp, SOLAR_MASS),
                "Rad": np.full(n_fp, SOLAR_RAD),
                "Teff": np.full(n_fp, SOLAR_TEFF),
                "logg": np.full(n_fp, SOLAR_LOGG),
                "rho": np.full(n_fp, 0.0),
                "E_rho": np.full(n_fp, rho_sd),
                "e_rho": np.full(n_fp, -rho_sd),
                "stellar_source": 2,
            })
            fallback_frames.append(fb3)
        dr25_ok = dr25_ok[have_dr25]
        if len(dr25_ok):
            matched = dr25_lookup.loc[dr25_ok["KIC"]]
            dens = matched["dens"].values
            rho_log = np.log10(dens / RHOS_GCM3)
            # dr25_full.csv (the archive's own DR25 stellar table, pulled
            # directly rather than the old stripped-down local
            # keplerstellar.csv extract) carries real per-star density
            # uncertainties: dens_err1 (always >=0, the upper delta) and
            # dens_err2 (always <=0, the lower delta -- so dens + dens_err2
            # is already the lower bound, not dens - dens_err2). Confirmed
            # against the actual column data (dens_err1 in [4.65e-8, 23.49],
            # dens_err2 in [-99.91, -9.67e-9], never the wrong sign) before
            # relying on this convention. Use these directly instead of the
            # order-unity placeholder this branch used to fall back on;
            # complete_kics above already required both to be present
            # (non-NaN), so no NaN reaches here. The 1e-6 g/cm^3 floor on the
            # lower bound is kept as a safety clamp for log10 (dens +
            # dens_err2 never actually goes non-positive in this file, but a
            # small fraction of rows come within ~1e-6 of it), matching the
            # same floor used in the Source_rowe==2 branch above.
            E_lin = matched["dens_err1"].values
            e_lin = matched["dens_err2"].values
            fb2 = pd.DataFrame({
                "KIC": dr25_ok["KIC"].values,
                "Mass": matched["mass"].values,
                "Rad": matched["radius"].values,
                "Teff": matched["teff"].values,
                "logg": matched["logg"].values,
                "rho": rho_log,
                "E_rho": np.log10((dens + E_lin) / RHOS_GCM3),
                "e_rho": np.log10(np.clip(dens + e_lin, 1e-6, None) / RHOS_GCM3),
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
        f"{(fallback_df['stellar_source']==1).sum()} DR25/dr25_full.csv, "
        f"{(fallback_df['stellar_source']==2).sum()} solar-with-Berger-scatter) -- "
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

        # T0_planet = stellar_df[stellar_df["KIC"]==row["kepid"]][]


        planet_star_radius_ratio = radius * RECM / (radius_star * RSCM)

        print(f"[rank {rank}] rho_star_true: ",rho_star_true)
        print(f"[rank {rank}] rho_star_uncertainty: ",rho_star_uncertainty)
        print(f"[rank {rank}] number of NaN in rho_star_true: ",np.sum(np.isnan(rho_star_true)))
        print(f"[rank {rank}] number of NaN in rho_star_uncertainty: ",np.sum(np.isnan(rho_star_uncertainty)))

        if row["koi_duration"] == 0 and row["koi_duration_err1"] == 0:
            # A handful of singles have koi_duration and its own reported
            # uncertainty both exactly 0.0 -- Rowe's table never fit a real
            # transit duration for these, and DR25 (checked in create_ksdc.py)
            # doesn't have them either. koi_duration==0 would otherwise feed
            # straight into a sin(0)==0 division-by-zero in this function's
            # "inside" formula, and there's no trustworthy central value to
            # build a posterior around anyway. Everything sampled above this
            # point (radius, period, b, mass, stellar radius/mass) is real
            # and independent of duration -- only eccentricity, omega, and
            # (derived below) inclination actually require this fit, so only
            # those are left as NaN for this planet.
            print(f"[rank {rank}] WARNING: kepid={row['kepid']} (singles index {index}) has "
                  f"koi_duration==0 with zero reported uncertainty and no DR25 fallback -- "
                  f"no real transit duration is available. Leaving eccentricity, omega, and "
                  f"inclination as NaN for this planet; radius, period, mass, and stellar "
                  f"radius/mass are still real sampled values.")
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


        # Remove false positives (and possibly other bad data ones)

        # Remove the planets with densities above or below a certain threshold, because they are unphysical
        print("length of df before requiring stability: ",len(df))
        if runprops["exclude_bad_densities"]:
            df = df[(df["rho_p"]<runprops["maximum_density"]) & (df["rho_p"]>runprops["minimum_density"])]
        print("length of df after excluding bad densities: ",len(df))


        print("length of df after before requiring stability: ",len(df))
        # Exclude any posterior draw that has a periastron less than 2 stellar radii
        # sm_axis = (df["Period_days"] * 24 * 60 * 60)**(2/3) * (G / (4*np.pi**2))**(1/3) * (df["M_pE"] * MEKG + stellar_df.set_index("KIC").loc[df["KIC"],"Mass"].values * MSKG)**(1/3)
        # periapsis = (1 - df["e"]) * sm_axis
        df = df[df['peri_R_s'] >= 2 * df['a_R_s']]
        print("length of df after after requiring stability: ",len(df))

        df = df[(df["Teff_rowe"] < 7000) & (df['Teff_rowe'] > 3000)]

        df = df[(df["logg_rowe"] > 4)]

        df = df[df["ecc_omega_convergence_failed"]==0]

        df = df[~(df["Status_rowe"][0] == 'F')] 


        # print("length of singles df after before requiring stability: ",len(processed_singles_dr_df))
        # sm_axis = (processed_singles_dr_df["Period_days"] * 24 * 60 * 60)**(2/3) * (G / (4*np.pi**2))**(1/3) * (processed_singles_dr_df["M_pE"] * MEKG + stellar_df.set_index("KIC").loc[processed_singles_dr_df["kepid"],"Mass"].values * MSKG)**(1/3)
        # periapsis = (1 - processed_singles_dr_df["e"]) * sm_axis
        # processed_singles_dr_df = processed_singles_dr_df[periapsis >= 2 * stellar_df.set_index("KIC").loc[processed_singles_dr_df["kepid"],"Rad"].values]
        # print("length of processed_singles_dr_df after requiring stability: ",len(processed_singles_dr_df))


        # Define the "unique planet" column which is the combo of the KIC and planet number
        # df['unique_planet'] = df['KIC'].astype(str) + "_" + df['planet'].astype(str)
        # Make a dict of how many unique planets there are, with the unique planet as the key and the count as the value
        kic_dict = df['kdc_id'].value_counts().to_dict()

        print("kic_dict: ", kic_dict)

        # List of unique planets that have less than 50 samples, which we remove because they will be improperly weighted
        unique_planet_to_remove = [k for k, v in kic_dict.items() if v < 50]

        # Actually remove these planets
        for unique_planet in unique_planet_to_remove:
            df = df[df['unique_planet'] != unique_planet]

        # Update the kic_dict_multis after removing the planets with less than 50 samples (necessary for reweighting procedure)
        kic_dict = {k: v for k, v in kic_dict.items() if v >= 50}

        print("kic_dict: ", kic_dict)
        
        # Give the RPMeoGrid the kic_dict so that it can reweight the planets
        voxel_grid.set_kic_dict(kic_dict)

        # Concatenate the multis and processed singles dfs together to get the KDC - Kepler Dynamical Catalog
        final_kg_kdc_df = df
        ######## ADD A FLAG TO SEE IF ITS A SINGLE OR A MULTI (FOR PLOTTING PURPOSES)

        # Where there doesn't exist a PhoDyMM value, fill with the best guess from the stellar catalog.
        # final_kdc_df["M_s"] = final_kdc_df['M_s'].fillna(final_kdc_df["KIC"].map(stellar_df.set_index("KIC")["Mass"]))
        # final_kdc_df["R_s"] = final_kdc_df['R_s'].fillna(final_kdc_df["KIC"].map(stellar_df.set_index("KIC")["Rad"]))

        # final_kdc_df["Teff"] = final_kdc_df["KIC"].map(stellar_df.set_index("KIC")["Teff"])


        print("final_kdc_df: ",final_kdc_df)

        print("final_kdc_df columns: ",final_kdc_df.columns)
        
        print("length of df after matching to stellar catalog, filtering densities: ",len(final_kdc_df))

        # Add the data to the RPMeoGrid voxel grid object (this object will be written to a json, then read in for the model runs)
        voxel_grid.add_data(final_kdc_df)

        # Create a small stellar df with 1000 random stars, to set up the completeness grid. (could be expanded to entire stellar catalog)
        stellar_df_reduced=stellar_df.sample(n=500,random_state=44)


    voxel_grid = comm.bcast(voxel_grid,root=0)
    stellar_df_reduced = comm.bcast(stellar_df_reduced,root=0)

    print("broadcasted voxel grid and stellar df")

    ## need to update setup completeness grid so that it has the right stellar column names (and uses berger stellar)
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

        final_kg_kdc_table = pa.Table.from_pandas(final_kg_kdc_df)
        csv.write_csv(final_kg_kdc_table, "../data/final_kg_kdc.csv")

        # final_kdc_df.to_csv("../data/final_kdc.csv")

        final_kg_kdc_df_columns = json.dumps(list(final_kg_kdc_df.columns))
        with open('../data/dataframe_column_names.json', "w") as f:
            f.write(final_kg_kdc_df_columns)
        
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