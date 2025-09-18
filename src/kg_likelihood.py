import os
import sys
import numpy as np
import time
from tqdm import tqdm
from mpi4py import MPI
from scipy.special import gamma, gammaln
from scipy.stats import norm, lognorm, uniform
from kg_priors import prior_args
from kg_constants import N_PHODYMM_STARS

from kg_probability_distributions import synthetic_catalog_to_grid, generate_catalog, get_probability_distributions, voxel_model_count

stellar_df = None
voxel_grid = None
# current_best_logL = None

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
            case "lambda_e":
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
                lp += uniform.logpdf(params[i], loc=mu, scale=sigma-mu)
            case _:
                raise ValueError(f"Unknown prior type: {type} for parameter {parameter_name}")
        
        if np.isnan(lp) or np.isinf(lp):
            return -np.inf
    # print("prior eval time: ",start_time-time.time())
    return lp


def parametric_log_likelihood(params):
    
    start_time = time.time()

    global voxel_grid, stellar_df

    rank = MPI.COMM_WORLD.Get_rank()
    # print(f"[log-prob on rank {rank}]", flush=True)
    # print(os.getpid())


    len_stellar_df = len(stellar_df)
    Gamma0 = params[0]
    grid_sum = 0.0
    p_Period, Period_fine_grid, p_mass, mass_fine_grid,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid, is_nan_in_pmfs, is_inf_in_pmfs = get_probability_distributions(params)
    
    # print("get probability distribution time is ", time.time() - start_time)


    if is_nan_in_pmfs: # If the pmfs are generated to contain NaN values, the parameters used to generate them are probably bad. Don't mess, just reject.
        print("nan in pmfs!")
        return -np.inf
    
    if is_inf_in_pmfs:
        print("inf in pmfs!")
        return -np.inf
    
    # pre_generation_time = time.time()
    synthetic_catalog = generate_catalog(stellar_df,p_Period, Period_fine_grid, p_mass, mass_fine_grid, γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid)
    # print("catalog generation time is ", time.time() - pre_generation_time)
    # time.sleep(5)

    method = "new faster way"

    if method == "new faster way":
    # puts the synthetic catalog into the voxel grid all at once.
        voxel_grid = synthetic_catalog_to_grid(synthetic_catalog,voxel_grid)

        voxel_num_data = voxel_grid.likelihood_array[:,:,:,:,:,0]
        model_count = voxel_grid.likelihood_array[:,:,:,:,:,1]

        print("voxel_num_data.shape",voxel_num_data.shape)
        print("model_count.shape",model_count.shape)

        print("num of voxel_num_data > 0:", len(voxel_num_data[voxel_num_data > 0]))

        print("voxel_num_data",np.ravel(voxel_num_data))
        # print("model_count",model_count)

        if np.any((voxel_num_data < 0) | (np.isnan(voxel_num_data))):
            print("aaaaa")
            return -np.inf
        elif np.any((model_count < 0) | (np.isnan(model_count))):
            print("aaaaaaaaaaa")
            return -np.inf
        

        zero_mask = (model_count == 0) & (voxel_num_data == 0)
        voxel_num_data = voxel_num_data[~zero_mask] # if both the model and data say there's nothing in a voxel, let's count it as a neutral contribution
        model_count = model_count[~zero_mask] 

        no_model_mask = (model_count == 0) & (voxel_num_data > 0)
        model_count[no_model_mask] = 1e-3

        # elif voxel_num_data < 0: # there should never be a negative voxel number of data... (could be a raise or warning statement?)
        #     print("negative voxel count!")
        #     return -np.inf
        # elif model_count < 0: # there should never be a negative model count number
        #     print("negative model count!")
        #     return -np.inf
        # elif model_count == 0 and voxel_num_data > 0: # if the model predicts nothing and the data has something, 
        #     # print("model=0, voxel > 0 !")
        #     model_count = 1e-8

            # grid_sum -= voxel_num_data * 10**5
            # continue

            # return -np.inf
        # elif np.isnan(model_count) or np.isnan(voxel_num_data): # if nans are somehow generated, the model should be rejected
        #     print("nan in voxel or model count!")
        #     return -np.inf
    
        grid_sum = (voxel_num_data * np.log(model_count) - model_count - gammaln(voxel_num_data+1))
        # print("grid_sum: ",grid_sum)
        grid_sum = np.sum(grid_sum)
        print("grid_sum after summing: ", grid_sum)

    else: 
        
        ########### old implementation
        pre_loop_time = time.time()
        total_model_count_time = 0
        voxel_number = 0 ###
        point_buffer = None

        for voxel in voxel_grid.voxel_array.flat:
            # print("likelihood df: ",voxel.df)
            # pre_data_count_time = time.time()
            voxel_num_data = len(voxel.df) / 1000 if not voxel.df.empty else 0  # preload the len(voxel.df)
            # print("data count time is ", time.time() - pre_data_count_time)
            # print("voxel: ", voxel)
            # print("voxel_num_data: ", voxel_num_data)
            # input()
            # model_count_time = time.time() ################
            model_count, point_buffer = voxel_model_count(voxel_grid,voxel,synthetic_catalog,point_buffer) # can we do this for all voxels all at once? build in weights
            # print("model count time is ", (model_count_time:=time.time() - model_count_time)) #############
            # total_model_count_time += model_count_time
            # print("model count: ", model_count)
            
            if model_count == 0 and voxel_num_data == 0: # if both the model and data say there's nothing in a voxel, let's count it as a neutral contribution
                voxel_number += 1
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
            
            pre_likelihood_function_time = time.time()
            grid_sum += (voxel_num_data * np.log(model_count) - model_count - gammaln(voxel_num_data+1))
            # print("likelihood function time is ", time.time() - pre_likelihood_function_time)
            
            if np.isnan(grid_sum) or np.isnan(voxel_num_data): # if we're still getting nans here, we are in some serious trouble...
                fail_number = "grid_sum" if np.isnan(grid_sum) else "voxel_num_data"
                print("model_count: ", model_count)
                print("voxel_num_data: ", voxel_num_data)
                print("voxel: ", voxel)
                raise ValueError(fail_number+" here is NaN, check your model or data!")
            
            voxel_number += 1

        print(f"{voxel_number} voxels evaluated")
    
    end_time = time.time()
    # print("total model count time is ", total_model_count_time)
    # print("average model count time is ", total_model_count_time / voxel_number)
    # print("the other thing ", )
    # print("loop eval time is ", end_time - pre_loop_time)
    if method == "new faster way": 
        print("1 eval time (new faster way) is ", end_time - start_time)
    # else:
        # print("1 eval time (old way) is ", end_time - start_time)
    # sys.exit(0)
    # print("evaluated normally")
    ##################### ^ old implementation

    logL = Gamma0 * grid_sum

    print("logL: ",logL)

    return logL if np.isfinite(logL) else -np.inf


def parametric_log_probability(params):

    prior = parametric_log_prior(params)

    return prior + parametric_log_likelihood(params) if np.isfinite(prior) else -np.inf