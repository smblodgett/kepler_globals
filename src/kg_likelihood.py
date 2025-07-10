import numpy as np
from scipy.special import gamma
from scipy.stats import norm, lognorm
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
    assert len(params) == len(prior_args.keys), "Number of parameters must match the number of priors!"
    lp = 0.0 
    for parameter_name, i in zip(prior_args.keys(), range(len(params))):
        mu, sigma, type = prior_args[parameter_name]
        
        print("length of params[i]: ", len(params[i]))
        input()
        
        if parameter_name == "C":
            print("C: ", params[i])
            if params[i] < 0:
                return -np.inf

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


def parametric_log_likelihood(params,voxel_grid,stellar_df):

    Gamma0 = params[0]
    grid_sum = 0.0
    p_Period, Period_fine_grid, p_mass, mass_fine_grid,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid = get_probability_distributions(params)
    synthetic_catalog = generate_catalog(stellar_df,p_Period, Period_fine_grid, p_mass, mass_fine_grid, γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid)
    for voxel in voxel_grid.voxel_array.flat:
        print("likelihood df: ",voxel.df)
        voxel_num_data = len(voxel.df) if not voxel.df.empty else 0
        # input()
        model_count = voxel_model_count(voxel_grid,voxel,synthetic_catalog)
        grid_sum += (model_count * np.log(voxel_num_data) - voxel_num_data - np.log(gamma(model_count+1)))
    return Gamma0 * grid_sum


def parametric_log_probability(params):

    prior = parametric_log_prior(params)
    return prior + parametric_log_likelihood(params) if prior != -np.inf else -np.inf