import numpy as np
from scipy.special import gamma
from scipy.stats import norm, lognorm
from kg_priors import prior_args
from kg_constants import N_PHODYMM_STARS

from kg_probability_distributions import voxel_model_count

def grid_log_probability(params,observed,N_HSU_STARS,observation_probability):
    R_mrp = params[0]
    if observed < 0 : print("warning! observed is:",observed)
    # observation_probability = max(1e-6, observation_probability) # 1 solution: fix observation probability to be a minimum value...
    
    if R_mrp < 0: # the only prior: our R_mrp must be positive
        return -np.inf
    return R_mrp * N_HSU_STARS * observation_probability * np.log(observed) - observed - np.log(gamma(R_mrp * N_HSU_STARS * observation_probability+1)) # test that this is maxed when expected == observed



def parametric_log_prior(params):
    assert len(params) == len(prior_args.keys), "Number of parameters must match the number of priors!"
    lp = 0.0 
    for parameter_name, i in zip(prior_args.keys(), range(len(params))):
        mu, sigma, type = prior_args[parameter_name]
        if type == "lnN":
            if params[i] <= 0:
                return -np.inf
            lp += lognorm.logpdf(params[i], s=sigma, scale=np.exp(mu))
        elif type == "N":
            lp += norm.logpdf(params[i], loc=mu, scale=sigma)
        else:
            raise ValueError(f"Unknown prior type: {type} for parameter {parameter_name}")
        
        if np.isnan(lp) or np.isinf(lp):
            return -np.inf
        
    return lp


def parametric_log_likelihood(params,voxel_grid):
    print("IN LLINKELIHOOODDDFSDFS")
    Gamma0 = params[0]
    grid_sum = 0.0
    for voxel in voxel_grid.voxel_array.flat:
        print("likelihood df: ",voxel.df)
        voxel_weighted_num_data = voxel.df["num_weighted_data"].iloc[0] if not voxel.df.empty else 0
        # input()
        model_count = N_PHODYMM_STARS * voxel_model_count(voxel,params)
        grid_sum += (model_count * np.log(voxel_weighted_num_data) - voxel_weighted_num_data - np.log(gamma(model_count+1)))
    return Gamma0 * grid_sum


def parametric_log_probability(params):

    return parametric_log_prior(params) + parametric_log_likelihood(params)