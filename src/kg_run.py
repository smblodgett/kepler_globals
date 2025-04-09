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
from kg_plots import *

# object to read in runprops file
class ReadJson:
    def __init__(self, filename):
        print('read the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data
    
    
def run_emcee(voxel_grid,voxel_id,runprops):
    
    # get specific voxel(s?)
    voxel = voxel_grid.find_voxel(voxel_id)
    voxel.create_initial_guess()
    print(voxel_id)
    print(voxel,voxel.initial_guess)
    assert type(voxel) == RPM_Voxel
    
    # create backend
    backend_folder = runprops["results_folder"] + "/backend/"
    os.makedirs(backend_folder, exist_ok=True)
    backend_filename = backend_folder + "voxel_" + str(voxel_id) + "_chain.h5"
    backend = emcee.backends.HDFBackend(backend_filename)
    backend.reset(runprops["nwalkers"], runprops["ndim"])
    
    # create sampler
    sampler = emcee.EnsembleSampler(runprops["nwalkers"], runprops["ndim"], 
    kg_likelihood.likelihood, backend=backend)
    
    print(np.sum(voxel.df['occurrence_rate_hsu']))
    print(voxel.initial_guess)
    print(np.max(np.abs([voxel.df['+sigma_hsu'],voxel.df['-sigma_hsu']])))
    print(voxel.initial_guess * (1 + np.random.normal(0,np.max(np.abs([voxel.df['+sigma_hsu'],voxel.df['-sigma_hsu']])))))
    
    p0 = np.array([[voxel.initial_guess * (1 + np.random.normal(0,np.max(np.abs([voxel.df['+sigma_hsu'],voxel.df['-sigma_hsu']]))))] for _ in range(runprops["nwalkers"])]) # take randomly from a normal distribution, choose the hsu error bounds for stdev...


    if runprops["verbose"]: print('sampler created. Starting the burn in.')
    
    if runprops['thin_run']:
        state = sampler.run_mcmc(p0, runprops["nburnin"], progress = True, store = True, thin=runprops["nthinning"])
    else:
        state = sampler.run_mcmc(p0, runprops["nburnin"], progress = True, store = True)
        
    
    
def main(voxel_id): 
    # verify path
    cwd = os.getcwd()
    print(cwd)        

    # find runprops path
    if 'src' in cwd:
        runprops_filename = "../runs/runprops.txt"
    elif 'runs' in cwd:
        runprops_filename = "runprops.txt"
    elif 'results' in cwd:
        runprops_filename = "runprops.txt"
    else:
        print('you are not starting from a proper directory. you should run kg_run.py from a src, runs, or a results directory.')
        sys.exit()

    # get runprops loaded in, find initial guess file
    getData = ReadJson(runprops_filename)
    runprops = getData.outProps()

    # create voxel grid data structure, load it with the data
    voxel_grid = RPM_Grid(radius_grid_array,period_grid_array,mass_grid_array)

    df = pd.read_csv(runprops["input_data_filename"]) 

    voxel_grid.setup_dataframes(df.columns)
    voxel_grid.add_data(df)

    if runprops["verbose"] : print(voxel_grid)
        
    run_emcee(voxel_grid,voxel_id,runprops)
    
    
if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("invalid input. Enter the voxel id to run kepler_globals on.")
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


# Now creating the sampler object

# backend_filename = runprops.get("results_folder")+ "/chain.h5"

# backend = emcee.backends.HDFBackend(backend_filename)
# backend.reset(nwalkers, ndim)
# moveset = [(emcee.moves.StretchMove(), 1.0),]




# sampler = emcee.EnsembleSampler(nwalkers, ndim, 
# kg_likelihood.probability, backend=backend,
#         args = (float_names, fixed_df, total_df_names, fit_scale, runprops, obsdf,geo_obj_pos, best_llhoods),
#         moves = moveset)

# if verbose: print('sampler created')

# nburnin = runprops.get("nburnin")
# nthinning = runprops.get("nthinning")
# if verbose:
#     print("Starting the burn in")
# if runprops.get('thin_run'):
#     state = sampler.run_mcmc(p0, nburnin, progress = True, store = True, thin=nthinning)
# else:
#     state = sampler.run_mcmc(p0, nburnin, progress = True, store = True)
    
    
# nsteps = runprops.get("nsteps")
# essgoal = runprops.get("essgoal")
# maxiter = runprops.get("maxiter")
# initsteps = runprops.get("nsteps")

# sampler,ess = mm_autorun.mm_autorun(sampler, essgoal, state, initsteps, maxiter, verbose, objname, p0, runprops)

# print("effective sample size = ", ess)
# chain = sampler.get_chain(thin = runprops.get("nthinning"))
# flatchain = sampler.get_chain(flat = True, thin = runprops.get("nthinning"))

# runpath = runprops.get("results_folder")+"/runprops.txt"
# runprops['total_df_names'] = runprops.get('total_df_names').tolist()
# del runprops['best_llhood']

# with open(runpath, 'w') as file:
#     file.write(json.dumps(runprops, indent = 4))

# recent_props = runprops.get("runs_file")+"/most_recent_runprops.txt"
# with open(recent_props, 'w') as file:
#     file.write(json.dumps(runprops, indent = 4))

# backendf = runprops.get('results_folder')+'chain.h5'
# backend = emcee.backends.HDFBackend(backendf)
# os.chdir(runprops.get("results_folder"))
