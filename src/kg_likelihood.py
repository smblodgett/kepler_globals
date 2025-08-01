import os
import numpy as np
import time
from tqdm import tqdm
from mpi4py import MPI
from scipy.special import gamma, gammaln
from scipy.stats import norm, lognorm, uniform
from kg_priors import prior_args
from kg_constants import N_PHODYMM_STARS

from kg_probability_distributions import voxel_model_count, generate_catalog, get_probability_distributions

stellar_df = None
voxel_grid = None

def grid_log_probability(params,observed,N_HSU_STARS,observation_probability):
    R_mrp = params[0]
    if observed < 0 : print("warning! observed is:",observed)
    # observation_probability = max(1e-6, observation_probability) # 1 solution: fix observation probability to be a minimum value...
    
    if R_mrp < 0: # the only prior: our R_mrp must be positive
        return -np.inf
    return R_mrp * N_HSU_STARS * observation_probability * np.log(observed) - observed - np.log(gamma(R_mrp * N_HSU_STARS * observation_probability+1)) # test that this is maxed when expected == observed



def parametric_log_prior(params):
    start_time = time.time()
    assert len(params) == len(prior_args.keys()), "Number of parameters must match the number of priors!"
    # print("params.shape: ", params.shape)
    lp = 0.0 
    for parameter_name, i in zip(prior_args.keys(), range(len(params))):
        mu, sigma, prior_type = prior_args[parameter_name]
        
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

        match prior_type:
            case "lnN":
                if params[i] <= 0:
                    return -np.inf
                lp += lognorm.logpdf(params[i], s=sigma, scale=np.exp(mu))
            case "N":
                lp += norm.logpdf(params[i], loc=mu, scale=sigma)
            case "U":
                lp += uniform.logpdf(params[i], loc=mu, scale=sigma-mu)
            case _:
                raise ValueError(f"Unknown prior type: {type} for parameter {parameter_name}")
        
        if np.isnan(lp) or np.isinf(lp):
            return -np.inf
    # print("prior eval time: ",start_time-time.time())
    return lp


def parametric_log_likelihood(params):

    global voxel_grid, stellar_df

    rank = MPI.COMM_WORLD.Get_rank()
    print(f"[log-prob on rank {rank}]", flush=True)
    print(os.getpid())


    start_time = time.time()
    len_stellar_df = len(stellar_df)
    Gamma0 = params[0]
    grid_sum = 0.0
    p_Period, Period_fine_grid, p_mass, mass_fine_grid,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid, is_nan_in_pmfs, is_inf_in_pmfs = get_probability_distributions(params)
    
    if is_nan_in_pmfs: # If the pmfs are generated to contain NaN values, the parameters used to generate them are probably bad. Don't mess, just reject.
        print("nan in pmfs!")
        return -np.inf
    
    if is_inf_in_pmfs:
        print("inf in pmfs!")
        return -np.inf
    
    synthetic_catalog_kde = generate_catalog(stellar_df,p_Period, Period_fine_grid, p_mass, mass_fine_grid, γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid)
    
    for voxel in voxel_grid.voxel_array.flat:
        # print("likelihood df: ",voxel.df)
        voxel_num_data = len(voxel.df) / 1000 if not voxel.df.empty else 0
        # print("voxel: ", voxel)
        # print("voxel_num_data: ", voxel_num_data)
        # input()
        model_count = voxel_model_count(voxel_grid,voxel,synthetic_catalog_kde,len_stellar_df)
        # print("model count: ", model_count)
        
        if model_count == 0 and voxel_num_data == 0: # if both the model and data say there's nothing in a voxel, let's count it as a neutral contribution
            continue
        elif voxel_num_data < 0: # there should never be a negative voxel number of data... (could be a raise or warning statement?)
            print("negative voxel count!")
            return -np.inf
        elif model_count < 0: # there should never be a negative model count number
            print("negative model count!")
            return -np.inf
        elif model_count == 0 and voxel_num_data > 0: # if the model predicts nothing and the data has something, 
            # print("model=0, voxel > 0 !")
            model_count = 1e-8

            # grid_sum -= voxel_num_data * 10**5
            # continue

            # return -np.inf
        elif np.isnan(model_count) or np.isnan(voxel_num_data): # if nans are somehow generated, the model should be rejected
            print("nan in voxel or model count!")
            return -np.inf
        
        # grid_sum += (model_count * np.log(voxel_num_data) - voxel_num_data - gammaln(model_count+1))
        grid_sum += (voxel_num_data * np.log(model_count) - model_count - gammaln(voxel_num_data+1))
        # print("grid sum: ",grid_sum)
        
        if np.isnan(grid_sum) or np.isnan(voxel_num_data): # if we're still getting nans here, we are in some serious trouble...
            fail_number = "grid_sum" if np.isnan(grid_sum) else "voxel_num_data"
            print("model_count: ", model_count)
            print("voxel_num_data: ", voxel_num_data)
            print("voxel: ", voxel)
            raise ValueError(fail_number+" here is NaN, check your model or data!")
        
        
    end_time = time.time()
    # print("evaluated normally")

    logL = Gamma0 * grid_sum
    return logL if np.isfinite(logL) else -np.inf


def parametric_log_probability(params):

    prior = parametric_log_prior(params)

    return prior + parametric_log_likelihood(params) if np.isfinite(prior) else -np.inf