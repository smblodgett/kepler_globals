import numpy as np
import time
import os
import json
from mpi4py import MPI
from scipy.special import gamma, gammaln, log_expit
from scipy.stats import norm, lognorm, uniform
from kg_priors import PriorArgs 
import matplotlib.pyplot as plt

from kg_constants import N_PHODYMM_SYSTEMS

from kg_probability_distributions import (
    synthetic_catalog_to_grid,
    synthetic_catalog_with_weights,
    generate_catalog,
    get_probability_distributions,
    joint_log_intrinsic_density,
    profile_optimal_gamma0,
)
from kg_utilities import density_given_mass_radius

stellar_info = None # this is a np array from the stellar_df that is defined and given cuts in kg_initialize_voxel_grid.py. Its length is the same as the synthetic catalog's
voxel_grid = None
# model_run_dir = None
model_id = None
density_prior_mask = None
# local_best_logProb = -np.inf
synthetic_multiplier = None
observed_catalog = None  # flat, unbinned real-catalog dict from kg_probability_distributions.load_flat_observed_catalog
likelihood_method = "pointprocess"  # "pointprocess" (new, unbinned) or "grid" (legacy 5-D histogram Poisson)
density_bounds = (0.01, 10.0)  # (min, max) physical density in g/cm^3, matching runprops minimum_density/maximum_density

prior_args = PriorArgs().load_priors()

def grid_log_probability(params,observed,N_HSU_STARS,observation_probability):
    R_mrp = params[0]
    if observed < 0 : print("warning! observed is:",observed)
    # observation_probability = max(1e-6, observation_probability) # 1 solution: fix observation probability to be a minimum value...
    
    if R_mrp < 0: # the only prior: our R_mrp must be positive
        return -np.inf
    return R_mrp * N_HSU_STARS * observation_probability * np.log(observed) - observed - np.log(gamma(R_mrp * N_HSU_STARS * observation_probability+1)) # test that this is maxed when expected == observed



def parametric_log_prior(params, model_id):

    priors = prior_args.get_priors(model_id)

    # start_time = time.time()
    assert len(params) == len(priors), f"Number of parameters must match the number of priors! Expected {len(priors)}, got {len(params)}"
    # print("params.shape: ", params.shape)
    lp = 0.0 
    for parameter_name, i in zip(priors, range(len(params))):
        mu, sigma, prior_type = parameter_name[1], parameter_name[2], parameter_name[3]
        
        # print("parameter_name: ", parameter_name)
        # print("params[i]: ", params[i])
        # print("length of params[i]: ", len(params[i]))
        # input()
        
        match parameter_name:
            case "C":
                if params[i] < 0:
                    return -np.inf
            case "mu_M":              ### the mean of the mass distribution shouldn't be beneath the lower mass limit of the catalog...
                if np.exp(params[i]) < 0.1:
                    return -np.inf
            case "lambda_e":
                if params[i] < 0:
                    return -np.inf
            case "Mbreak1":
                if params[i] > params[i+1]: # if Mbreak1 is greater than Mbreak2
                    return -np.inf
                if params[i] < 0:
                    return -np.inf
            case "Mbreak2":
                if params[i] < 0:
                    return -np.inf
            

        match prior_type:
            case "lnN":
                if params[i] <= 0:
                    return -np.inf
                lp += lognorm.logpdf(params[i], s=sigma, scale=np.exp(mu))
            case "N":
                lp += norm.logpdf(params[i], loc=mu, scale=sigma)
            case "U":
                # print("prior matched: U")
                # print("mu: ",mu)
                # print("sigma: ",sigma)
                # print("parameter name: ",parameter_name)
                # # lp += uniform.logpdf(params[i], loc=mu, scale=sigma-mu)
                if mu > params[i] or sigma < params[i]:
                    lp = -np.inf
                # print("lp: ", lp)
            case _:
                raise ValueError(f"Unknown prior type: {type} for parameter {parameter_name}")
        
        if np.isnan(lp) or np.isinf(lp):
            return -np.inf
    # print("prior eval time: ",start_time-time.time())
    return lp


def parametric_log_likelihood_pointprocess(params, model_id, min_density=None, max_density=None):
    """
    Unbinned / point-process ("extended maximum likelihood") formulation of
    the likelihood. See the large comment block in kg_probability_distributions.py
    ("Semi-analytical (unbinned/point-process) likelihood machinery") for the
    full statistical justification.

    Real data are never put into a histogram. Every real planet's posterior
    draws get evaluated at their own exact (period, mass, radius, e, omega):
    analytically for the intrinsic population density (joint_log_intrinsic_density)
    and via the interpolated completeness/transit-probability grids for
    selection effects. The only Monte Carlo piece left is the total expected
    number of detections (Lambda_hat), which has no closed form because
    completeness doesn't -- that's the "hybrid" of analytic + numerical this
    project's notes asked for.

    Following Neil & Rogers (2020)'s correction of Foreman-Mackey et al. (2014):
    the two selection-effect factors are NOT interchangeable and must not both
    appear at the data points. p_det (pipeline detection efficiency) only
    belongs in the Lambda_hat integral -- for a planet already in the catalog,
    multiplying by p_det again double-conditions on an event (its own
    detection) that's already certain (Loredo 2004; Mandel et al. 2019). p_tr
    (geometric transit probability) legitimately belongs in *both* places,
    since the transiting subset is the only population we can ever detect at
    all. So: Lambda_hat uses the full completeness (p_det*p_tr, via
    interpolate_completeness), while the per-planet data term uses p_tr alone
    (via interpolate_transit_probability).

    Gamma0 (the overall rate normalization) is NOT one of `params` and is not
    sampled by the MCMC. It's profiled out analytically every call: for fixed
    shape parameters, logL is maximized over Gamma0 at
    Gamma0_opt = N_obs / Lambda_tilde (Lambda_tilde being Lambda_hat at
    Gamma0=1), and substituting that back in gives the profile likelihood
    actually returned here (see profile_optimal_gamma0 and the
    "Semi-analytical..." comment block in kg_probability_distributions.py for
    the full derivation). This also returns lambda_tilde itself (as part of
    the blobs, alongside the rng metadata) so that Gamma0's own posterior can
    be reconstructed after the fact -- see kg_plots.pointprocess_gamma0_posterior_plot.

    logL_profile = N_obs*log(N_obs) - N_obs*log(Lambda_tilde) - N_obs
                   + sum_j log(<f_pop(theta_j) * p_tr(theta_j)>_j)
    """
    start_time = time.time()

    global voxel_grid, stellar_info, synthetic_multiplier, observed_catalog, density_bounds

    rank = MPI.COMM_WORLD.Get_rank()

    min_density = density_bounds[0] if min_density is None else min_density
    max_density = density_bounds[1] if max_density is None else max_density

    (p_Period, Period_fine_grid, p_mass, mass_fine_grid, γ0, γ1, γ2, mass_break_1, mass_break_2,
     σ0, σ1, σ2, C, p_ecc, eccentricity_fine_grid,
     is_nan_in_pmfs, is_inf_in_pmfs, is_neg_in_pmfs) = get_probability_distributions(params)

    print(f"rank {rank} get probability distribution time is ", (prob_dist_time := time.time()) - start_time, flush=True)

    if is_nan_in_pmfs or is_inf_in_pmfs or is_neg_in_pmfs:
        return -np.inf, {"master_seed": -1, "rank_seed": -1, "time_seed": -1}, rank, np.nan

    synthetic_catalog, rng_metadata = generate_catalog(
        stellar_info, p_Period, Period_fine_grid, p_mass, mass_fine_grid,
        γ0, γ1, γ2, mass_break_1, mass_break_2, σ0, σ1, σ2, C, p_ecc, eccentricity_fine_grid, rank
    )

    print(f"rank {rank} generate catalog time is ", (gen_cat_time := time.time()) - prob_dist_time, flush=True)

    # ---- Lambda_tilde: Monte Carlo estimate of the total expected number of
    # detections at Gamma0=1 (the shape-only part of Lambda_hat) ----
    # synthetic_catalog_with_weights rearranges to (radius, period, mass, e, omega),
    # clips to the grid's coordinate bounds, drops dynamically-implausible orbits
    # (periapsis within 2 stellar radii), and returns completeness evaluated
    # pointwise (interpolated, no histogram involved).
    trimmed_catalog, completeness_weights, _ = synthetic_catalog_with_weights(synthetic_catalog, voxel_grid, stellar_info)

    synth_density = density_given_mass_radius(trimmed_catalog[:, 2], trimmed_catalog[:, 0])
    density_mask = (synth_density >= min_density) & (synth_density <= max_density)
    # Keep Lambda_tilde on the same physically-plausible density domain as the
    # data (final_kdc.csv is already filtered to this range), so Gamma0_opt
    # isn't biased by synthetic planets that could never appear in the data.

    Lambda_tilde = np.sum(completeness_weights[density_mask]) / synthetic_multiplier

    print(f"rank {rank} lambda_tilde calc time is ", (lambda_time := time.time()) - gen_cat_time, flush=True)
    print(f"rank {rank} Lambda_tilde: {Lambda_tilde}, n synthetic kept: {np.sum(density_mask)} / {len(synthetic_catalog)}", flush=True)

    if not np.isfinite(Lambda_tilde) or Lambda_tilde <= 0:
        return -np.inf, rng_metadata, rank, np.nan

    # ---- data term: evaluate every real posterior draw at its own exact location ----
    obs = observed_catalog
    log_f_obs = joint_log_intrinsic_density(params, obs["P"], obs["M"], obs["R"], obs["e"], obs["omega"])

    obs_points = np.column_stack([obs["R"], obs["P"], obs["M"], obs["e"], obs["omega"]])  # (radius, period, mass, e, omega) order
    # p_tr only -- NOT the combined completeness -- per Neil & Rogers (2020): these
    # are already-confirmed detections, so re-multiplying by p_det here would
    # double-condition on their detection (see the docstring above).

    # in more understandable words, these HAVE ALREADY BEEN DETECTED
    # so p_det is already 1 for them, and we can't multiply by p_det again...that would double-count it!
    transit_prob_obs = voxel_grid.interpolate_transit_probability(obs_points) #

    ALPHA = 1e-300
    log_transit_prob_obs = np.log(np.maximum(transit_prob_obs, ALPHA))

    vals = log_f_obs + log_transit_prob_obs
    vals = np.where(np.isfinite(vals), vals, -700.0)  # floor rather than -inf so reduceat stays well-behaved

    seg_starts = obs["seg_starts"]
    seg_counts = obs["seg_counts"]

    # Grouped (per-planet) log-mean-exp over that planet's posterior draws,
    # fully vectorized via reduceat since draws are pre-sorted by planet.
    seg_max = np.maximum.reduceat(vals, seg_starts)
    shifted = vals - np.repeat(seg_max, seg_counts)
    seg_sumexp = np.add.reduceat(np.exp(shifted), seg_starts)
    term_per_planet = seg_max + np.log(seg_sumexp) - np.log(seg_counts)

    n_planets = obs["n_planets"]
    Gamma0_opt = profile_optimal_gamma0(n_planets, Lambda_tilde)

    # logL at Gamma0_opt: n_planets*log(Gamma0_opt) - Gamma0_opt*Lambda_tilde
    # simplifies (since Gamma0_opt*Lambda_tilde == n_planets exactly) to
    # n_planets*log(Gamma0_opt) - n_planets; the n_planets*log(n_planets)
    # part of that log is a fixed constant (same every step), so it's
    # included here just for scale-consistency with the old Gamma0-sampled
    # logL values, not because it affects the MCMC in any way.
    logL_data = n_planets * np.log(Gamma0_opt) + np.sum(term_per_planet)
    logL = logL_data - n_planets

    print(f"rank {rank} data term calc time is ", (time.time() - lambda_time), flush=True)
    print(f"rank {rank} total eval time is ", (time.time() - start_time), flush=True)
    print(f"rank {rank} logL: {logL} (data term: {logL_data}, Lambda_tilde: {Lambda_tilde}, Gamma0_opt: {Gamma0_opt})", flush=True)

    return (logL if np.isfinite(logL) else -np.inf, rng_metadata, rank, Lambda_tilde)


def parametric_log_likelihood(params, model_id):
    """Dispatches to the point-process (default) or legacy grid likelihood,
    controlled by the module-level `likelihood_method` global (set from
    kg_run_param.py via runprops["likelihood_method"]). Both variants return
    a (logL, rng_metadata, rank, lambda_tilde) 4-tuple -- lambda_tilde is the
    shape-only (Gamma0=1) expected-count integral each one profiled Gamma0
    out of, kept so it can be stored in the emcee blobs and used later to
    reconstruct Gamma0's posterior (see profile_optimal_gamma0 and
    kg_plots.pointprocess_gamma0_posterior_plot)."""
    global likelihood_method
    if likelihood_method == "grid":
        return parametric_log_likelihood_grid(params, model_id)
    return parametric_log_likelihood_pointprocess(params, model_id)


def parametric_log_likelihood_grid(params, model_id):
    """Legacy 5-D binned-histogram Poisson likelihood (the original
    implementation). Kept for A/B comparison against
    parametric_log_likelihood_pointprocess -- switch via
    runprops["likelihood_method"] = "grid"."""

    start_time = time.time()

    global voxel_grid, stellar_info, synthetic_multiplier

    # print("len(stellar_df): ", len(stellar_df))

    rank = MPI.COMM_WORLD.Get_rank()

    with open('/proc/loadavg') as f:
        load = f.read().split()[0:3]
    print(f"[rank {rank}] host_load: {load}", flush=True)

    
    # print(f"[log-prob on rank {rank}]", flush=True)
    # print(os.getpid())

    ######################################### to do monday: testing on why gen catalog is so slow! and other speedups if possible 
    ######################################### to do monday: also look at graphs, analyze performance of new model 


    # len_stellar_df = len(stellar_df)
    # print("params: ", params)

    grid_sum = 0.0
    p_Period, Period_fine_grid, p_mass, mass_fine_grid,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid, is_nan_in_pmfs, is_inf_in_pmfs, is_neg_in_pmfs = get_probability_distributions(params)

    # print(params)
    print(f"rank {rank} get probability distribution time is ", (prob_dist_time:=time.time()) - start_time,flush=True)


    if is_nan_in_pmfs: # If the pmfs are generated to contain NaN values, the parameters used to generate them are probably bad. Don't mess, just reject.
        # print("nan in pmfs!")
        return -np.inf, {"master_seed": -1, "rank_seed": -1, "time_seed": -1}, rank, np.nan

    if is_inf_in_pmfs:
        # print("inf in pmfs!")
        return -np.inf, {"master_seed": -1, "rank_seed": -1, "time_seed": -1}, rank, np.nan

    if is_neg_in_pmfs:
        # print("negative values in pmfs!")
        return -np.inf, {"master_seed": -1, "rank_seed": -1, "time_seed": -1}, rank, np.nan

    synthetic_catalog, rng_metadata = generate_catalog(stellar_info,p_Period, Period_fine_grid, p_mass, mass_fine_grid, γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid,rank)
    ######## implement making sure that the random generated one

    print(f"rank {rank} generate catalog time is ", (gen_cat_time:=time.time()) - prob_dist_time, flush=True)


    ######################### TO DO: MAKE SURE DATA IS IN PLANETS, NOT POSTERIOR DRAWS
    local_voxel_grid = synthetic_catalog_to_grid(synthetic_catalog,voxel_grid,stellar_info,synthetic_multiplier)

    print(f"rank {rank} catalog to grid time is ", (cat_grid_time:=time.time()) - gen_cat_time, flush=True)


    voxel_num_data = local_voxel_grid.likelihood_array[:,:,:,:,:,0]
    # This is the Gamma0=1 model histogram (completeness-weighted, unscaled) --
    # Gamma0 is no longer a sampled parameter, so it's profiled out below
    # exactly the same way as in parametric_log_likelihood_pointprocess.
    voxel_hist = local_voxel_grid.likelihood_array[:,:,:,:,:,1]
    print("total data count: ", np.sum(voxel_num_data))
    print("total model count (Gamma0=1): ", np.sum(voxel_hist))



    if np.any((voxel_num_data < 0) | (np.isnan(voxel_num_data))):
        print("aaaaa")
        return -np.inf, rng_metadata, rank, np.nan
    elif np.any((voxel_hist < 0) | (np.isnan(voxel_hist))):
        print("aaaaaaaaaaa")
        return -np.inf, rng_metadata, rank, np.nan


    # yes_data_yes_model_voxels = (voxel_num_data > 0) & (voxel_hist > 0)
    # yes_data_no_model_voxels = (voxel_num_data > 0) & (voxel_hist == 0)
    # no_data_yes_model_voxels = (voxel_num_data == 0) & (voxel_hist > 0)
    # no_data_no_model_voxels = (voxel_num_data == 0) & (voxel_hist == 0)
    # print("yes data yes model: ", np.sum(yes_data_yes_model_voxels),"yes data no model: ", np.sum(yes_data_no_model_voxels),"no data yes model: ", np.sum(no_data_yes_model_voxels),"no data no model: ", np.sum(no_data_no_model_voxels))

    # zero-ness doesn't depend on Gamma0 (as long as Gamma0 > 0 finite), so
    # this mask can be built directly from the unscaled hist/data, before
    # Gamma0_opt is even known.
    zero_mask = (voxel_hist == 0) & (voxel_num_data == 0)
    mask = ~zero_mask & density_prior_mask

    # ---- profile out Gamma0: same closed-form trick as the point-process
    # likelihood (profile_optimal_gamma0), applied to the grid's Poisson
    # sum instead of the unbinned data term. Ignoring the ALPHA floor below
    # (which only binds for hist==0 voxels that still have data -- a fixed,
    # Gamma0-independent penalty already handled by those voxels being kept
    # in `mask`), d(logL)/dGamma0 = sum(data)/Gamma0 - sum(hist) = 0 gives
    # Gamma0_opt = sum(data[mask]) / sum(hist[mask]).
    Lambda_tilde = np.sum(voxel_hist[mask])
    n_data = np.sum(voxel_num_data[mask])
    Gamma0 = profile_optimal_gamma0(n_data, Lambda_tilde)
    if not np.isfinite(Gamma0) or Gamma0 < 0:
        return -np.inf, rng_metadata, rank, Lambda_tilde

    model_count = Gamma0 * voxel_hist

        # Poisson branch — evaluated on ALL voxels in density_prior_mask, smoothed to avoid log(0)
    ALPHA = 1e-8
    model_count_floored = np.maximum(model_count[mask], ALPHA)
    voxel_num_data_all = voxel_num_data[mask]

    logL_i = (voxel_num_data_all * np.log(model_count_floored)
            - model_count_floored
            - gammaln(voxel_num_data_all + 1))

    logL = np.sum(logL_i) 


    print(f"rank {rank} mask and sum time is ", (mask_sum_time:=time.time()) - cat_grid_time, flush=True)
# # 
    print(f"rank {rank} total eval time is ", (time.time() - start_time), flush = True)

#     print("logL: ",logL,flush=True)

    ########## testing stuff


    return (logL if np.isfinite(logL) else -np.inf, rng_metadata, rank, Lambda_tilde)



def parametric_log_probability(params):

    # global model_run_dir
    # global local_best_logProb
    global model_id

    prior = parametric_log_prior(params,model_id)

    if not np.isfinite(prior):
        # print("prior is not finite with this params!!!")
        # print("params: ", params)
        return -np.inf , -1, -1, -1, np.nan

    logL, rng_metadata, rank, lambda_tilde = parametric_log_likelihood(params,model_id)

    logProb = prior + logL


    return logProb, rng_metadata['master_seed'], rng_metadata['rank_seed'], rng_metadata['time_seed'], lambda_tilde