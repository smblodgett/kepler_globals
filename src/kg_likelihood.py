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

from kg_probability_distributions import synthetic_catalog_to_grid, generate_catalog, get_probability_distributions

stellar_info = None # this is a np array from the stellar_df that is defined and given cuts in kg_initialize_voxel_grid.py. Its length is the same as the synthetic catalog's
voxel_grid = None
model_run_dir = None
model_id = None
density_prior_mask = None
# local_best_logProb = -np.inf
synthetic_multiplier = None

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

    Gamma0 = 10**params[0]
    grid_sum = 0.0
    p_Period, Period_fine_grid, p_mass, mass_fine_grid,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid, is_nan_in_pmfs, is_inf_in_pmfs, is_neg_in_pmfs = get_probability_distributions(params)

    # print(params)
    print(f"rank {rank} get probability distribution time is ", (prob_dist_time:=time.time()) - start_time,flush=True)


    if is_nan_in_pmfs: # If the pmfs are generated to contain NaN values, the parameters used to generate them are probably bad. Don't mess, just reject.
        # print("nan in pmfs!")
        return -np.inf, {"master_seed": -1, "rank_seed": -1, "time_seed": -1}, rank
    
    if is_inf_in_pmfs:
        # print("inf in pmfs!")
        return -np.inf, {"master_seed": -1, "rank_seed": -1, "time_seed": -1}, rank

    if is_neg_in_pmfs:
        # print("negative values in pmfs!")
        return -np.inf, {"master_seed": -1, "rank_seed": -1, "time_seed": -1}, rank
    
    synthetic_catalog, rng_metadata = generate_catalog(stellar_info,p_Period, Period_fine_grid, p_mass, mass_fine_grid, γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid,rank)
    ######## implement making sure that the random generated one 

    print(f"rank {rank} generate catalog time is ", (gen_cat_time:=time.time()) - prob_dist_time, flush=True)


    ######################### TO DO: MAKE SURE DATA IS IN PLANETS, NOT POSTERIOR DRAWS
    local_voxel_grid = synthetic_catalog_to_grid(synthetic_catalog,voxel_grid,synthetic_multiplier)

    print(f"rank {rank} catalog to grid time is ", (cat_grid_time:=time.time()) - gen_cat_time, flush=True)


    voxel_num_data = local_voxel_grid.likelihood_array[:,:,:,:,:,0]
    print("total data count (before multiplying by Gamma0): ", np.sum(local_voxel_grid.likelihood_array[:,:,:,:,:,0]))
    print("total model count (before multiplying by Gamma0): ", np.sum(local_voxel_grid.likelihood_array[:,:,:,:,:,1]))
    model_count = Gamma0 * local_voxel_grid.likelihood_array[:,:,:,:,:,1]



    if np.any((voxel_num_data < 0) | (np.isnan(voxel_num_data))):
        print("aaaaa")
        return -np.inf, rng_metadata, rank
    elif np.any((model_count < 0) | (np.isnan(model_count))):
        print("aaaaaaaaaaa")
        return -np.inf, rng_metadata, rank
    

    # yes_data_yes_model_voxels = (voxel_num_data > 0) & (model_count > 0)
    # yes_data_no_model_voxels = (voxel_num_data > 0) & (model_count == 0)
    # no_data_yes_model_voxels = (voxel_num_data == 0) & (model_count > 0)
    # no_data_no_model_voxels = (voxel_num_data == 0) & (model_count == 0)
    # print("yes data yes model: ", np.sum(yes_data_yes_model_voxels),"yes data no model: ", np.sum(yes_data_no_model_voxels),"no data yes model: ", np.sum(no_data_yes_model_voxels),"no data no model: ", np.sum(no_data_no_model_voxels))


    zero_mask = (model_count == 0) & (voxel_num_data == 0)
    # no_model_mask = (model_count == 0) & (voxel_num_data > 0)
    mask = ~zero_mask  & density_prior_mask

        # Poisson branch — evaluated on ALL voxels in density_prior_mask, smoothed to avoid log(0)
    ALPHA = 1e-8
    mask = ~zero_mask & density_prior_mask
    model_count_floored = np.maximum(model_count[mask], ALPHA)
    voxel_num_data_all = voxel_num_data[mask]

    logL_i = (voxel_num_data_all * np.log(model_count_floored)
            - model_count_floored
            - gammaln(voxel_num_data_all + 1))


    # log_norm_const = np.log1p(-np.exp(-10 ** params[18]))
    # logL_noise_i = log_norm_const - 10 ** params[18] * voxel_num_data_all

    # # Per-voxel mixture, NOT a weighted sum of sums

    # log_pi = log_expit(params[19])            # log(sigmoid(x)), numerically stable
    # log_1m_pi = log_expit(-params[19])

    # logL_i = np.logaddexp(log_1m_pi + logL_poisson_i, log_pi + logL_noise_i)

    logL = np.sum(logL_i) 

    # combined_poisson_mask =  density_prior_mask & ~zero_mask & ~no_model_mask
    # combined_noise_mask = density_prior_mask 
    # # print("shape of mask: ", combined_mask.shape)

    # voxel_num_data_poisson = voxel_num_data[combined_poisson_mask] # if both the model and data say there's nothing in a voxel, let's count it as a neutral contribution
    # model_count_poisson = model_count[combined_poisson_mask] 

    # voxel_num_data_noise = voxel_num_data[combined_noise_mask]
    # # no_model_mask = (model_count == 0) & (voxel_num_data > 0)

    # ##### print out # of yes data/yes model, yes data/no model, no data/yes model, no/no 
    # grid_sum_poisson = (voxel_num_data_poisson * np.log(model_count_poisson) - model_count_poisson - gammaln(voxel_num_data_poisson+1))

    # # find contribution of log-likelihood of typical voxel
    # # median_logP_contribution = np.median(grid_sum_poisson)

    # # penalize the no model, yes data case by the typical log likelihood times however much data is there
    # # this HAS to be a fixed value (otherwise MCMC will just trade off between this likelihood and the Poisson)
    # grid_sum_noise = params[18] * voxel_num_data_noise

    # # apply to grid_sum
    # # print("grid_sum: ",grid_sum)
    # logL = np.sum((1-params[19])* grid_sum_poisson) + np.sum(params[19] * grid_sum_noise)

    print(f"rank {rank} mask and sum time is ", (mask_sum_time:=time.time()) - cat_grid_time, flush=True)
# # 
    print(f"rank {rank} total eval time is ", (time.time() - start_time), flush = True)

#     print("logL: ",logL,flush=True)




    ########## testing stuff

    # if total_grid_sum > best_total_grid_sum:
    #     my_grid_sum = grid_sum
    #     logL = total_grid_sum
    #     best_total_grid_sum = total_grid_sum
    #     best_my_model_count = model_count
    #     best_my_tau = test_threshold
    
    # if model_id == 0:
    #     model_count += 1e-7
    # else:
    #     model_count += 10 ** params[18] 

    # print("voxel_num_data.shape post-mask: ", voxel_num_data.shape)
    # print("model_count.shape post-mask: ", model_count.shape)

    # # model_count = model_count[density_prior_mask]

    # print("my sum(model_count): ",np.sum(model_count))
    # print("my sum(voxel_num_data): ",np.sum(voxel_num_data))

    # my_model_count = model_count.copy()
    # my_synthetic_catalog = synthetic_catalog.copy()
    # my_combined_mask = combined_mask.copy()


    # grid_sum = (voxel_num_data * np.log(model_count) - model_count - gammaln(voxel_num_data+1))

    # my_grid_sum = grid_sum.copy()
    # ##### histogram of grid_sum (seeing if like 10 voxels are controlling everything)

    # # print("grid_sum: ",grid_sum)
    # total_grid_sum = np.sum(grid_sum)
    # print("grid_sum after summing: ", total_grid_sum)
    
    # end_time = time.time()

    # logL = total_grid_sum



    ################################## comparing if Neil and Rogers period is better fit than ours
    # neil_rogers_params = params.copy()
    # neil_rogers_params[13] = -0.76
    # Gamma0 = 0.89
    # p_Period, Period_fine_grid, p_mass, mass_fine_grid,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid, is_nan_in_pmfs, is_inf_in_pmfs, is_neg_in_pmfs = get_probability_distributions(neil_rogers_params)
    # synthetic_catalog, rng_metadata = generate_catalog(stellar_df,p_Period, Period_fine_grid, p_mass, mass_fine_grid, γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid,rank)
    # print("synthetic_catalog head: ", synthetic_catalog[:5])
    # print("shape of synthetic_catalog: ", synthetic_catalog.shape)
    # rogers_voxel_grid = synthetic_catalog_to_grid(synthetic_catalog,voxel_grid)
    # voxel_num_data = rogers_voxel_grid.likelihood_array[:,:,:,:,:,0]
    # model_count = Gamma0 * rogers_voxel_grid.likelihood_array[:,:,:,:,:,1]
    # zero_mask = (model_count == 0) & (voxel_num_data == 0)
    # combined_mask = ~zero_mask & density_prior_mask
    # print("shape of mask: ", combined_mask.shape)
    # voxel_num_data = voxel_num_data[combined_mask] # if both the model and data say there's nothing in a voxel, let's count it as a neutral contribution
    # nr_model_count = model_count[combined_mask] 
    # best_neil_rogers_total_grid_sum = -np.inf

    # for test_threshold in [-21,-17,-13,-10,-7,-4]:
    #     neil_rogers_params[18] = test_threshold
    #     if model_id == 0:
    #         model_count += 1e-7
    #     else:
    #         model_count = nr_model_count + 10 ** neil_rogers_params[18] 
    #     print("voxel_num_data_rogers.shape post-mask: ", voxel_num_data.shape)
    #     print("model_count.shape_rogers post-mask: ", model_count.shape)
    #     # model_count = model_count[density_prior_mask]

    #     grid_sum = (voxel_num_data * np.log(model_count) - model_count - gammaln(voxel_num_data+1))
    #     # print("grid_sum: ",grid_sum)
    #     total_grid_sum = np.sum(grid_sum)
    #     if total_grid_sum > best_neil_rogers_total_grid_sum:
    #         rogers_grid_sum = grid_sum
    #         logL_rogers = total_grid_sum
    #         best_neil_rogers_total_grid_sum = total_grid_sum
    #         best_nr_model_count = model_count
    #         best_tau = test_threshold

    #     print("grid_sum after summing: ", total_grid_sum)
    #     print("logL: ", logL, "logL_rogers: ", logL_rogers)


    # print("sum(model_count_rogers): ",np.sum(model_count))
    # print("sum(voxel_num_data_rogers): ",np.sum(voxel_num_data))
    # if np.random.random() < 0.005:


    #     plt.hist(my_grid_sum.flatten(),bins=75)
    #     plt.xlabel("logL")
    #     plt.yscale('log')
    #     plt.savefig("grid_sum.png")
    #     plt.close()

      


    #     period_param_grid_array = [0.2,0.75,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,12.0,14.0,16.0,20.0,24.0,28.0,32.0,40.0,48.0,64.0,128.0,192.0,256.0,360.0,500.0]

    #     original_shape = combined_mask.shape

    #     reconstructed_my_grid_sum = np.zeros(original_shape)
    #     reconstructed_my_grid_sum[my_combined_mask] = my_grid_sum
    #     count_my_grid_sum_period = np.sum(reconstructed_my_grid_sum, axis=(0,2,3,4))
    #     reconstructed_nr_grid_sum = np.zeros(original_shape)
    #     reconstructed_nr_grid_sum[combined_mask] = rogers_grid_sum
    #     count_nr_grid_sum_period = np.sum(reconstructed_nr_grid_sum, axis=(0,2,3,4))

    #     edges = np.asarray(period_param_grid_array)
    #     centers = 0.5*(edges[:-1] + edges[1:])
    #     widths = 1
    #     x = np.arange(len(centers))

    #     plt.figure(dpi=150, facecolor='w')
    #     plt.bar(x, -1*count_my_grid_sum_period, width=widths, alpha=0.5, label='my grid sum')
    #     plt.bar(x, -1*count_nr_grid_sum_period, width=widths, alpha=0.5, label='nr grid sum')
        
    #     edge_positions = np.arange(len(edges)) - 0.5
    #     plt.xticks(edge_positions, [f"{e:.2f}" for e in edges], rotation=45)
    #     plt.legend()
    #     plt.xlabel("grid sum *-1 by period")
    #     plt.savefig("grid_sum_compare.png")
    #     plt.close()

    #     assert original_shape == voxel_grid.likelihood_array[:,:,:,:,:,0].shape, "The original shape of the data and model arrays should match the shape of the combined mask."

    #     reconstructed_nr_model = np.zeros(original_shape)
    #     reconstructed_my_model = np.zeros(original_shape)
    #     reconstructed_data = np.zeros(original_shape)


    #     reconstructed_my_model[my_combined_mask] = best_my_model_count
    #     reconstructed_nr_model[combined_mask] = best_nr_model_count
    #     reconstructed_data[combined_mask] = voxel_num_data

    #     data_count_period = np.sum(reconstructed_data, axis=(0,2,3,4))
    #     my_model_count_period = np.sum(reconstructed_my_model, axis=(0,2,3,4))
    #     nr_model_count_period = np.sum(reconstructed_nr_model, axis=(0,2,3,4))
    #     my_physical_catalog_count_period, _ = np.histogram(my_synthetic_catalog[:,0],bins=period_param_grid_array)
    #     nr_physical_catalog_count_period, _ = np.histogram(synthetic_catalog[:,0],bins=period_param_grid_array)
        


    #     ### SHOULD BE IN TERMS OF PLANETS, NOT POSTERIOR DRAWS
    #     plt.figure(dpi=200, facecolor='w')
    #     plt.bar(x, data_count_period, width=widths, alpha=0.5, label='data')
    #     plt.bar(x, my_model_count_period, width=widths, alpha=0.5, label='inferred observed catalog') 
    #     plt.bar(x, nr_model_count_period, width=widths, alpha=0.5, label='N&R observed catalog') 
    #     plt.bar(x, nr_physical_catalog_count_period, width=widths, alpha=0.5, label='N&R physical catalog')
    #     plt.bar(x, my_physical_catalog_count_period, width=widths, alpha=0.5, label='my physical catalog') 

        
    #     edge_positions = np.arange(len(edges)) - 0.5

    #     plt.xticks(edge_positions, [f"{e:.2f}" for e in edges], rotation=45)

    #     plt.xlabel(rf"P- $\tau=${best_my_tau:.1f} my $\beta_1=${params[12]:.2f} $\beta_2=${params[13]:.2f} Pbreak={params[14]:.2f} Gamma={10**params[0]:.2f} : \n nr $\beta_1=${neil_rogers_params[12]:.2f} $\beta_2=${neil_rogers_params[13]:.2f} Pbreak={neil_rogers_params[14]:.2f} Gamma={Gamma0:.1f} $\tau=${best_tau:.1f}",fontsize=5)
    #     plt.yscale('log')
    #     plt.legend()
    #     plt.title(f'period:: logL N&R - {logL_rogers:.2f} :: logL mine - {logL:.2f}')
    #     plt.tight_layout()
    #     plt.savefig(f'model_period_test.png')
    #     plt.close()
    #     raise ValueError()

        ################ TESTING GRAPHS


    return (logL if np.isfinite(logL) else -np.inf, rng_metadata, rank)



def parametric_log_probability(params):

    global model_run_dir
    # global local_best_logProb
    global model_id

    prior = parametric_log_prior(params,model_id)

    if not np.isfinite(prior):
        # print("prior is not finite with this params!!!")
        # print("params: ", params)
        return -np.inf , -1, -1, -1

    logL, rng_metadata, rank = parametric_log_likelihood(params,model_id)

    # print("rng_metadata: ", rng_metadata,flush=True)
    # print("prior: ",prior,flush=True)


    logProb = prior + logL

    # rng_metadata |= {"logProb":logProb}


    # if logProb > local_best_logProb:
    #     local_best_logProb = logProb
        
        # os.makedirs(model_run_dir+"/rank_metadata",exist_ok=True)
        # with open(model_run_dir+f"/rank_metadata/{rank}.json", "w") as f:
        #     json.dump(rng_metadata,f)
        
    # if logProb == -np.inf:
        # print("logProb is -inf with this params!!!")
        # print("params: ", params)
        # print("prior: ", prior)
        # print("logL: ", logL)

    return logProb, rng_metadata['master_seed'], rng_metadata['rank_seed'], rng_metadata['time_seed']