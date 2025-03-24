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

from kg_griddefiner import *
from kg_grid_boundary_arrays import radius_grid_array, period_grid_array, mass_grid_array

# object to read in runprops file
class ReadJson:
    def __init__(self, filename):
        print('Read the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data
    
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
    print('You are not starting from a proper directory. You should run kg_run.py from a src, runs, or a results directory.')
    sys.exit()
    
# get runprops loaded in, find initial guess file
getData = ReadJson(runprops_filename)
runprops = getData.outProps()
init_filename = runprops.get("init_filename")

# get the other parameters from runprops
verbose = runprops.get("verbose")
object_name = runprops.get("object_name")
nwalkers = runprops.get("nwalkers")
nsteps = runprops.get("nsteps")
nburnin = runprops.get("nburnin")
make_plots = runprops.get("make_plots")
output_filename = runprops.get("output_filename")
input_data_filename = runprops.get("input_data_filename")


# create voxel grid data structure, load it with the data
voxel_grid = RPM_Grid(radius_grid_array,period_grid_array,mass_grid_array)

df = pd.read_csv(input_data_filename)

voxel_grid.setup_dataframes(df.columns)
voxel_grid.add_data(df)

if verbose: print(voxel_grid)
    
    
voxel_grid.plot()


    
    
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
