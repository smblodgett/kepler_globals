import numpy as np
import time
import os
import json
from mpi4py import MPI
from scipy.special import gamma, gammaln
from scipy.stats import norm, lognorm, uniform
from kg_priors import PriorArgs 

from kg_constants import N_PHODYMM_SYSTEMS

from kg_probability_distributions import synthetic_catalog_to_grid, generate_catalog, get_probability_distributions, voxel_model_count

stellar_df = None
voxel_grid = None
model_run_dir = None
model_id = None
density_prior_mask = None
local_best_logProb = -np.inf

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
    assert len(params) == len(priors), "Number of parameters must match the number of priors!"
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


def parametric_log_likelihood(params, model_id):
    
    start_time = time.time()

    global voxel_grid, stellar_df

    print("len(stellar_df): ", len(stellar_df))

    rank = MPI.COMM_WORLD.Get_rank()
    # print(f"[log-prob on rank {rank}]", flush=True)
    # print(os.getpid())


    # len_stellar_df = len(stellar_df)
    print("params: ", params)

    Gamma0 = 10**params[0]
    grid_sum = 0.0
    p_Period, Period_fine_grid, p_mass, mass_fine_grid,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid, is_nan_in_pmfs, is_inf_in_pmfs, is_neg_in_pmfs = get_probability_distributions(params)
    
    # print(params)
    # print("get probability distribution time is ", time.time() - start_time)


    if is_nan_in_pmfs: # If the pmfs are generated to contain NaN values, the parameters used to generate them are probably bad. Don't mess, just reject.
        print("nan in pmfs!")
        return -np.inf, {}, rank
    
    if is_inf_in_pmfs:
        print("inf in pmfs!")
        return -np.inf, {}, rank
    
    if is_neg_in_pmfs:
        print("negative values in pmfs!")
        return -np.inf, {}, rank
    
    synthetic_catalog, rng_metadata = generate_catalog(stellar_df,p_Period, Period_fine_grid, p_mass, mass_fine_grid, γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid,rank)
    ######## implement making sure that the random generated one 

    print("synthetic_catalog head: ", synthetic_catalog[:5])
    print("shape of synthetic_catalog: ", synthetic_catalog.shape)

    ### TO DO: MAKE SURE DATA IS IN PLANETS, NOT POSTERIOR DRAWS
    local_voxel_grid = synthetic_catalog_to_grid(synthetic_catalog,voxel_grid)

    voxel_num_data = local_voxel_grid.likelihood_array[:,:,:,:,:,0]       # chat is saying to scale the model count by the number of phodymm systems that we have...? could this be right?
    model_count = Gamma0 * local_voxel_grid.likelihood_array[:,:,:,:,:,1] 

    # print("voxel_num_data.shape: ",voxel_num_data.shape)
    # print("model_count.shape: ",model_count.shape)

    #### If a voxel is outside of the priors of the data, we should exclude it



    # print("num of voxel_num_data > 0:", len(voxel_num_data[voxel_num_data > 0]))
    # print("num of model_count > 0:", len(model_count[model_count > 0]))
    # print("shape of voxel_num_data > 0:", voxel_num_data[voxel_num_data > 0].shape)
    # print("shape of model_count > 0:", model_count[model_count > 0].shape)
    # print("num of voxel_num_data > 1:", len(voxel_num_data[voxel_num_data > 1]))
    # print("num of model_count > 1:", len(model_count[model_count > 1]))

    if np.any((voxel_num_data < 0) | (np.isnan(voxel_num_data))):
        print("aaaaa")
        return -np.inf, rng_metadata, rank
    elif np.any((model_count < 0) | (np.isnan(model_count))):
        print("aaaaaaaaaaa")
        return -np.inf, rng_metadata, rank
    

    zero_mask = (model_count == 0) & (voxel_num_data == 0)
    voxel_num_data = voxel_num_data[~zero_mask] # if both the model and data say there's nothing in a voxel, let's count it as a neutral contribution
    model_count = model_count[~zero_mask] 


    # no_model_mask = (model_count == 0) & (voxel_num_data > 0)
    
    if model_id == 0:
        model_count += 1e-7
    else:
        model_count += 10 ** params[18] 

    model_count = model_count[density_prior_mask]

    print("sum(model_count): ",np.sum(model_count))
    print("sum(voxel_num_data): ",np.sum(voxel_num_data))


    grid_sum = (voxel_num_data * np.log(model_count) - model_count - gammaln(voxel_num_data+1))
    # print("grid_sum: ",grid_sum)
    total_grid_sum = np.sum(grid_sum)
    print("grid_sum after summing: ", total_grid_sum)
    
    end_time = time.time()

    logL = total_grid_sum

    print("logL: ",logL,flush=True)

    return (logL if np.isfinite(logL) else -np.inf, rng_metadata, rank)


def parametric_log_probability(params):

    global model_run_dir
    global local_best_logProb
    global model_id

    prior = parametric_log_prior(params,model_id)

    if not np.isfinite(prior):
        print("prior is not finite with this params!!!")
        print("params: ", params)
        return -np.inf

    logL, rng_metadata, rank = parametric_log_likelihood(params,model_id)

    # print("prior: ",prior,flush=True)


    logProb = prior + logL if np.isfinite(prior) else -np.inf

    rng_metadata |= {"logProb":logProb}


    if logProb > local_best_logProb:
        local_best_logProb = logProb
        
        os.makedirs(model_run_dir+"/rank_metadata",exist_ok=True)
        with open(model_run_dir+f"/rank_metadata/{rank}.json", "w") as f:
            json.dump(rng_metadata,f)
        
    if logProb == -np.inf:
        print("logProb is -inf with this params!!!")
        print("params: ", params)
        print("prior: ", prior)
        print("logL: ", logL)

    return logProb