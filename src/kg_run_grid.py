# main script that runs Kepler Globals modeling: as of now, this is modeling the voxels of the M-P-R relationship,
# assuming each voxel's independence

# begun October 3, 2024
# developed by Steven Blodgett, with Darin Ragozzine, Dallin Spencer, and Daniel Jones
# codebase drawn from Dallin Spencer's multi_moon


import pandas as pd
import numpy as np
import sys
import os
import emcee
import time

from kg_likelihood import grid_log_probability
from kg_griddefiner import *
from kg_constants import N_HSU_STARS
from kg_grid_boundary_arrays import radius_grid_array, period_grid_array, mass_grid_array
from kg_utilities import radius_given_density_mass, mass_given_density_radius, ReadJson
    

def timer(is_timer,benchmark_message_string,mode='benchmark'):
    """Tracks and prints the runtime of the script since script start."""
    global old_time
    global start_time
    if mode == 'final' and is_timer:
        end_time = time.time()
        run_time = end_time - start_time
        print(f"total runtime: {run_time} s")

    if mode == 'benchmark'and is_timer:    
        new_time = time.time()
        benchmark = new_time - old_time
        print("time since "+benchmark_message_string+f": {benchmark} s")
        old_time = new_time
    

def number_of_singles_in_voxel(voxel, expanded_dr_df):
    """Returns the number of singles in a given voxel."""
    mask = ((expanded_dr_df["radius"] < voxel.top_radius) &
            (expanded_dr_df["radius"] > voxel.bottom_radius) &
            (expanded_dr_df["period"] < voxel.top_period) &
            (expanded_dr_df["period"] > voxel.bottom_period) &
            (expanded_dr_df["mass"] < voxel.top_mass) &
            (expanded_dr_df["mass"] > voxel.bottom_mass)
            )
    print("all singles length: ", len(expanded_dr_df))
    print("singles length: ",len(expanded_dr_df.loc[mask]))
    return len(expanded_dr_df.loc[mask])


def run_emcee(voxel_grid,voxel_id,runprops,dr_path="../data/q1_q17_dr25.csv",expanded_dr_path="../data/expanded_dr25_singles.csv",hsu_star_path="../data/hsu_stellar_catalog_output.csv"):
    """Configures and runs the emcee MCMC sampler."""
    # Get the specific voxel to run the model on. 
    voxel = voxel_grid.find_voxel_by_id(voxel_id)
    assert type(voxel) == RPMVoxel
    print(voxel)
    timer(runprops["timer"],"voxel find")

    # If the voxel is completely outside of the density prior, just skip it.
    if voxel.is_implausible(runprops["upper_rho_prior"],runprops["lower_rho_prior"]):
        if not runprops["suppress_warnings"]:
            print("This voxel is outside of the density priors. emcee will not be run on it.")
            with open(runprops["log_filename"], "a") as file:
                file.write("Voxel outside density priors for: "+str(voxel_id)+"\n")
        sys.exit(1)

    # Get DR25 catalog.
    dr_df = pd.read_csv(dr_path)
    # Get processed singles dataframe. 
    expanded_dr_singles_df = pd.read_csv(expanded_dr_path) # These have already been filtered for the Hsu stellar catalog (and given the proper values)
    # Get Hsu stellar catalog.
    hsu_star_df = pd.read_csv(hsu_star_path)
    # Mark the DR25 catalog with those that are in the Hsu stellar catalog.
    dr_df["in_hsu"] = dr_df["kepid"].isin(hsu_star_df["kepid"]).astype(int)
    
    timer(runprops["timer"],"other readin")

    
    # Create the emcee backend.
    match runprops["upper_rho_prior"], runprops["uniform_densities"]:
        case 30, False:
            backend_folder = runprops["results_folder"] + "/backend_30/"
        case 30, True:
            backend_folder = runprops["results_folder"] + "/backend_30_uniform/"
        case 10, False:
            backend_folder = runprops["results_folder"] + "/backend_10/"
        case 10, True:
            backend_folder = runprops["results_folder"] + "/backend_10_uniform/"
        case _, True | False:
            raise ValueError("Invalid upper_rho_prior. Must be 10 or 30 g/cm^3.")

    os.makedirs(backend_folder, exist_ok=True)
    backend_filename = backend_folder + "voxel_" + str(voxel_id) + "_chain.h5"
    if os.path.exists(backend_filename):
        os.remove(backend_filename)
    backend = emcee.backends.HDFBackend(backend_filename)
    backend.reset(runprops["nwalkers"], runprops["ndim"])
    # Only use planets that are in the DR25 and in Hsu's stellar catalog. 
    phodymm_dr_df = voxel.df[voxel.df["Status_rowe"].str[2]=='P']
    phodymm_dr_df = phodymm_dr_df[phodymm_dr_df["hsu_flag"]==1]
    # The actual number of PhoDyMM observed planets (the number of posterior draws/rows in the voxel's dataframe) plus the number of singles, divided by 1000 (to turn posterior draws into planets)
    actual_phodymm_observed = (len(phodymm_dr_df) + number_of_singles_in_voxel(voxel,expanded_dr_singles_df)) / 1000  
    
    # Calculate the number of planets that the Hsu model says are in the RP pixel.
    observed_planets_num = 0 
    for index, row in dr_df[dr_df["in_hsu"] == 1].iterrows():
        if row["koi_prad"] < voxel.top_radius and row["koi_prad"] > voxel.bottom_radius and row["koi_period"] < voxel.top_period and row["koi_period"] > voxel.bottom_period:
            observed_planets_num += 1
    
    # Calculate the observation probability, aka the fraction of planets of this pixel that are actually observed based off of the given Hsu occurrence rates. 
    hsu_occurrence_rate = voxel_grid.get_occurrence_rate(voxel_id)

    observation_probability = observed_planets_num / (hsu_occurrence_rate * N_HSU_STARS)
    if observation_probability == 0:
        with open(runprops["log_filename"], "a") as file:
            file.write("observation probability = 0 for: "+str(voxel_id)+f" (expected obs: {hsu_occurrence_rate*N_HSU_STARS})"+str(voxel)+"\n")
        if not runprops["suppress_warnings"]: 
            print("Observation probability = 0. Not running emcee.") # what about observation probability?
        sys.exit(1)

    # Find how many planets we should actually expect to observe based off of the Hsu model.
    if len(voxel.df) != 0: 
        R_mrp_initial_guess = voxel.df["mass_divided_weights"].iloc[0] # this is Rmrp's initial guess
    else:
        R_mrp_initial_guess = 0.0

    if R_mrp_initial_guess == 0 and actual_phodymm_observed == 0: # seems like if BOTH have nothing, we should just...not run emcee on it
        with open(runprops["log_filename"], "a") as file:
                file.write("R_mrp and D_mrp = 0 for: "+str(voxel_id)+"\n")
        if not runprops["suppress_warnings"]: 
            print("Both the R_mrp and actual_PhoDyMM values = 0. Not running emcee.") # what about observation probability?
        sys.exit(1)

    # Create the emcee sampler.
    sampler = emcee.EnsembleSampler(runprops["nwalkers"], runprops["ndim"], 
    grid_log_probability, backend=backend, args=(actual_phodymm_observed,N_HSU_STARS,observation_probability))

    timer(runprops["timer"],"emcee setup")

    ### pass Rmrp into the likelihood function, not Mmrp
    if R_mrp_initial_guess == 0:
        p0 = np.array([[R_mrp_initial_guess + (np.random.normal(1e-5,1e-7))] for _ in range(runprops["nwalkers"])]) # take randomly from a normal distribution, choose the hsu error bounds for stdev... #### this probably needs to be changed based off of what the expected should actually be??
    else:
        p0 = np.array([[R_mrp_initial_guess + (np.random.normal(0,R_mrp_initial_guess/10))] for _ in range(runprops["nwalkers"])]) # take randomly from a normal distribution, choose the hsu error bounds for stdev... #### this probably needs to be changed based off of what the expected should actually be??


# correct for the distribution of 
    
    if runprops["verbose"]: print('sampler created. Beginning run.')
    
    if runprops['thin_run']:
        state = sampler.run_mcmc(p0, runprops['nburnin']+runprops["nsteps"], progress = True, store = True, thin=runprops["nthinning"])
    else:
        state = sampler.run_mcmc(p0, runprops['nburnin']+runprops["nsteps"], progress = True, store = True)

    with open(runprops["log_filename"], "a") as file:
        file.write("success: "+str(voxel_id)+"\n")
    timer(runprops["timer"],"emcee run")
    

def main(voxel_id): 

    # Verify the correct path script is being run from. 
    cwd = os.getcwd()
    print(cwd)        

    # Find the runprops file path. 
    if 'src' in cwd:
        runprops_filename = "../runs/grid_runprops.txt"
    else:
        print('you are not starting from a proper directory. you should run kg_run_grid.py from the src directory.')
        sys.exit(1)
    
    # Get runprops loaded in, find the initial guess file.
    getData = ReadJson(runprops_filename)
    runprops = getData.outProps()

    timer(runprops["timer"],"start (& runprops read)")

    # Create voxel grid data structure.
    voxel_grid = RPMGrid(radius_grid_array,period_grid_array,mass_grid_array)
    
    use_cache = os.path.isdir(runprops["voxel_data_folder"]) and not runprops["reload_KMDC"]

    if not runprops["suppress_warnings"]:
        if not use_cache:
            print("Warning! use_cache is",use_cache,"meaning that this run will take a long time!")
            print("Only run this way if your voxel data hasn't yet been cached.")
    
    # If the voxels don't have their data cached, then read in everything.
    if not use_cache:
        df = pd.read_csv(runprops["input_data_filename"],index_col=0)
        if runprops["verbose"]: print("read in the catalog without accessing the cache")
    # Otherwise, you can just read in 1 voxel that has its data cached.    
    else:
        df = pd.read_csv(runprops["voxel_data_folder"]+"/voxel_"+str(voxel_id)+".csv",index_col=0)
        if runprops["verbose"]: print("read in cached df")
    
    if runprops["uniform_densities"]:
        uniform_densities = np.random.uniform(runprops["lower_rho_prior"],runprops["upper_rho_prior"],len(df))
        df["M_pJ"] = mass_given_density_radius(uniform_densities,df["R_pJ"])
    
    timer(runprops["timer"],"data readin")

    # Setup and load grid with data. If data is not cached, then cache data from whole grid into voxel dataframes.
    voxel_grid.setup_dataframes(df.columns,voxel_id,use_cache)
    
    timer(runprops["timer"],"setup dataframes")

    voxel_grid.add_data(df)

    timer(runprops["timer"],"add data")

    if not use_cache:
        os.makedirs(runprops["voxel_data_folder"],exist_ok=True)
        voxel_grid.cache_dataframes(runprops["voxel_data_folder"])
        if not runprops["uniform_densities"]: 
            voxel_grid.cache_prior_excluded_values(runprops["voxel_data_folder"],runprops["upper_rho_prior"],runprops["lower_rho_prior"])

    timer(runprops["timer"],"cache dataframes")

    # Partition the Hsu et al. weights by mass.
    voxel_grid.make_mass_divided_weights(voxel_id,runprops["voxel_data_folder"],use_cache,runprops["upper_rho_prior"],runprops["lower_rho_prior"])

    timer(runprops["timer"],"weight partition")

    if runprops["upper_rho_prior"] == 30:
        expanded_dr_path="../data/expanded_dr25_singles_30_001.csv"
    elif runprops["upper_rho_prior"] == 10:
        if runprops["verbose"]: print("You are using 10 g/cm3 density prior!")
        expanded_dr_path="../data/expanded_dr25_singles_10_001.csv"

    try:        
        run_emcee(voxel_grid,voxel_id,runprops,expanded_dr_path=expanded_dr_path)
        sys.exit(0)
    except Exception as e:
        print("Error occurred..." + str(e))
        with open(runprops["log_filename"], "a") as file:
            file.write(str(e)+": "+str(voxel_id)+"\n")
    finally:
        timer(runprops["timer"],"",mode="final")
    


if __name__ == "__main__":
    old_time = time.time()
    start_time = old_time
    if len(sys.argv) != 2:
        print("invalid input. Enter which voxel id you want to run kepler_globals on.")
        sys.exit(1)
        
    voxel_id = int(sys.argv[1])
    main(voxel_id)
        
    

# figure out a way to visualize this...maybe show it before setup? then again after doing emcee? figure 

# run MCMC on each cell in the voxel grid:
# any voxel that is outside of the density prior should not even be run through emcee...
# if we have an empty voxel, then skip and output a list of voxels, flagging empty ones based on why they're empty
# for the impossibly heavy planets, we could absolutely put a prior for density