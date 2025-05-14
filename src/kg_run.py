# main script that runs Kepler Globals modeling: as of now, this is modeling the voxels of the M-P-R relationship,
# assuming each voxel's independence

# begun October 3, 2024
# developed by Steven Blodgett, with Darin Ragozzine, Dallin Spencer, and Daniel Jones
# codebase drawn from Dallin Spencer's multi_moon


import commentjson as json
import pandas as pd
import numpy as np
import sys
import os
import emcee

import kg_likelihood
from kg_griddefiner import *
from kg_grid_boundary_arrays import radius_grid_array, period_grid_array, mass_grid_array

N_HSU_STARS = 80006 


class ReadJson:
    """Read and store the contents of a Json file in a dict."""
    def __init__(self, filename):
        """Load the Json file."""
        print('reading in the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        """Return the parsed Json dictionary."""
        return self.data
    
    
def run_emcee(voxel_grid,voxel_id,runprops,dr_path="../data/q1_q17_dr25.csv",hsu_star_path="../data/hsu_stellar_catalog_output.csv"):
    
    # Get DR25 catalog.
    dr_df = pd.read_csv(dr_path)
    
    hsu_star_df = pd.read_csv(hsu_star_path)
    
    dr_df["in_hsu"] = dr_df["kepid"].isin(hsu_star_df["kepid"]).astype(int)
    
    # Get specific voxel to run model on. 
    voxel = voxel_grid.find_voxel_by_id(voxel_id)
#     voxel.create_initial_guess()
    print(voxel_id)
    print(voxel,voxel.initial_guess)
    assert type(voxel) == RPMVoxel
    
    # create backend
    backend_folder = runprops["results_folder"] + "/backend/"
    os.makedirs(backend_folder, exist_ok=True)
    backend_filename = backend_folder + "voxel_" + str(voxel_id) + "_chain.h5"
    backend = emcee.backends.HDFBackend(backend_filename)
    backend.reset(runprops["nwalkers"], runprops["ndim"])
    
    phodymm_dr_df = voxel.df[voxel.df["Status_rowe"].str[2]=='P']
    
    actual_obs = len(phodymm_dr_df) / 1000 
    
    # Create the emcee sampler.
    sampler = emcee.EnsembleSampler(runprops["nwalkers"], runprops["ndim"], 
    kg_likelihood.likelihood, backend=backend, args=(actual_obs,))
    

    observed_planets_num = 0 
    for index, row in dr_df[dr_df["in_hsu"] == 1].iterrows():
        if row["koi_prad"] < voxel.top_radius and row["koi_prad"] > voxel.bottom_radius and row["koi_period"] < voxel.top_period and row["koi_period"] > voxel.bottom_period:
            observed_planets_num += 1
    
    observation_probability = observed_planets_num / (voxel.df["occurrence_rate_hsu"].iloc[0] * N_HSU_STARS)
    
    expected = N_HSU_STARS * voxel.df["mass_divided_weights"].iloc[0] * observation_probability
    
    p0 = np.array([[expected * (np.random.lognormal(0,np.max(np.abs([voxel.df['+sigma_hsu']/10,voxel.df['-sigma_hsu']/10]))))] for _ in range(runprops["nwalkers"])]) # take randomly from a normal distribution, choose the hsu error bounds for stdev... #### this probably needs to be changed based off of what the expected should actually be

# put things in units of planets
# correct for the distribution of 
    
    if runprops["verbose"]: print('sampler created. Starting the burn in.')
    
    if runprops['thin_run']:
        state = sampler.run_mcmc(p0, runprops["nsteps"], progress = True, store = True, thin=runprops["nthinning"])
    else:
        state = sampler.run_mcmc(p0, runprops["nsteps"], progress = True, store = True)
        
    
    
def main(voxel_id): 
    # Verify the correct path script is being run from. 
    cwd = os.getcwd()
    print(cwd)        

    # Find the runprops file path. 
    if 'src' in cwd:
        runprops_filename = "../runs/runprops.txt"
    elif 'runs' in cwd:
        runprops_filename = "runprops.txt"
    elif 'results' in cwd:
        runprops_filename = "runprops.txt"
    else:
        print('you are not starting from a proper directory. you should run kg_run.py from a src, runs, or a results directory.')
        sys.exit()

    # Get runprops loaded in, find the initial guess file.
    getData = ReadJson(runprops_filename)
    runprops = getData.outProps()

    # Create voxel grid data structure, load it with the data.
    voxel_grid = RPMGrid(radius_grid_array,period_grid_array,mass_grid_array)

    df = pd.read_csv(runprops["input_data_filename"]) 

    voxel_grid.setup_dataframes(df.columns)
    voxel_grid.add_data(df)
    
#     print("column points in 0.5-0.75 radius, 4-8 days:",voxel_grid.count_points_in_RP_column(0.75,0.5,8.0,4.0))
#     print("points in 0.5-0.75 radius, 4-8 days, 0-.25 M:",voxel_grid.find_voxel_by_coordinates(0.55,5,0.15))
#     print("points in 0.5-0.75 radius, 4-8 days, .25-.5 M:",voxel_grid.find_voxel_by_coordinates(0.55,5,0.35))
#     print("points in 0.5-0.75 radius, 4-8 days, .5-.75 M:",voxel_grid.find_voxel_by_coordinates(0.55,5,0.65))
#     print("points in 0.5-0.75 radius, 4-8 days, .75-1 M:",voxel_grid.find_voxel_by_coordinates(0.55,5,0.85))
#     print("points in 0.5-0.75 radius, 4-8 days, 1-1.5 M:",voxel_grid.find_voxel_by_coordinates(0.55,5,1.15))
#     print("points in 0.5-0.75 radius, 4-8 days, 1.5-2 M:",voxel_grid.find_voxel_by_coordinates(0.55,5,1.75))
#     print("points in 0.5-0.75 radius, 4-8 days, 2-3 M:",voxel_grid.find_voxel_by_coordinates(0.55,5,2.75))
#     print("points in 0.5-0.75 radius, 4-8 days, 3-4 M:",voxel_grid.find_voxel_by_coordinates(0.55,5,3.75))
#     print("points in 0.5-0.75 radius, 4-8 days, 4-6 M:",voxel_grid.find_voxel_by_coordinates(0.55,5,4.75))

#     input()

    voxel_grid.make_mass_divided_weights()

    if runprops["verbose"] : print(voxel_grid)
        
    run_emcee(voxel_grid,voxel_id,runprops)
    
    
if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("invalid input. Enter which voxel id you want to run kepler_globals on.")
        sys.exit()
        
    voxel_id = int(sys.argv[1])
    main(voxel_id)
        

    
# initial guess: the sum of the occurrence weights for each datapoint in a voxel

# figure out a way to visualize this...maybe show it before setup? then again after doing emcee? figure 
### number of posterior draws in each voxel (multiplied by occurrence ratings)
# get a bin number for each voxel (could be a key for the initial guess .csv)

# a runprop that specifies "all" vs 1 particular voxel

# maybe parallelizing over voxels? ask Dallin about this!
# supercomputer parallizes by jobs, even if that takes multiple cores

# run MCMC on each cell in the voxel grid:
# any voxel that is outside of the density prior should not even be run through emcee...
# if we have an empty voxel, then skip and output a list of voxels, flagging empty ones based on why they're empty
# for the impossibly heavy planets, we could absolutely put a prior for density

# figure out how to do a virtual environment: how to set up? could keep dependencies safe n'stuff (this might be important for running on the supercomputer)
