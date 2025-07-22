import numpy as np
import time
from scipy.special import gammaln
from scipy.stats import norm, lognorm, uniform
from kg_priors import prior_args
from kg_constants import N_PHODYMM_STARS

from kg_probability_distributions import voxel_model_count, generate_catalog, get_probability_distributions

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
    print("params.shape: ", params.shape)
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


def parametric_log_likelihood(params,voxel_grid,stellar_df):
    start_time = time.time()
    Gamma0 = params[0]
    grid_sum = 0.0
    p_Period, Period_fine_grid, p_mass, mass_fine_grid,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid = get_probability_distributions(params)
    synthetic_catalog = generate_catalog(stellar_df,p_Period, Period_fine_grid, p_mass, mass_fine_grid, γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid)
    for voxel in voxel_grid.voxel_array.flat:
        # print("likelihood df: ",voxel.df)
        voxel_num_data = len(voxel.df) / 1000 if not voxel.df.empty else 0
        # print("voxel_num_data: ", voxel_num_data)
        # input()
        model_count = voxel_model_count(voxel_grid,voxel,synthetic_catalog)
        
        if model_count <= 0 and voxel_num_data <= 0:
            continue
        elif voxel_num_data <= 0:
            return -np.inf
        elif model_count < 0:
            return -np.inf
        elif np.isnan(model_count):
            return -np.inf
        
        grid_sum += (model_count * np.log(voxel_num_data) - voxel_num_data - gammaln(model_count+1))
        
        if np.isnan(grid_sum) or np.isnan(voxel_num_data):
            fail_number = "grid_sum" if np.isnan(grid_sum) else "voxel_num_data"
            print("model_count: ", model_count)
            print("voxel_num_data: ", voxel_num_data)
            print("voxel: ", voxel)
            raise ValueError(fail_number+" here is NaN, check your model or data!")
        
        
    end_time = time.time()
    # print("log_likelihood eval time: ",end_time-start_time)
    logL = Gamma0 * grid_sum
    return logL if np.isfinite(logL) else -np.inf


def parametric_log_probability(params,voxel_grid,stellar_df):

    prior = parametric_log_prior(params)

    return prior + parametric_log_likelihood(params,voxel_grid,stellar_df) if prior != -np.inf else -np.inf