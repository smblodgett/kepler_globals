"""
The main script to run the MCMC sampling of the hierarchical model parameters.
begun October 3, 2024
developed by Steven Blodgett, with the advisement of Darin Ragozzine, Dallin Spencer, and Daniel Jones
codebase drawn from Dallin Spencer's multi_moon
"""


# initial necessary imports
from mpi4py import MPI
import os

# rank initialization signatures
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
print(f"[Rank {rank}/{size}] starting up")
print(os.system("hostname"))

# space out the walkers by a tenth of a second
import time
time.sleep(.005*rank) 

import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys
import emcee
from datetime import datetime
import json
from schwimmbad import MPIPool

import kg_likelihood
from kg_griddefiner import *
from kg_param_initial_guess import get_initial_guess
from kg_utilities import ReadJson, density_given_mass_radius
from kg_plots import MES_grid_plot
from kg_grid_object_hook import grid_object_hook
from kg_param_boundary_arrays import radius_grid_array, period_grid_array, mass_grid_array, eccentricity_grid_array, omega_grid_array

print(f"[Rank {rank}/{size}] finished imports")
print(os.system("hostname"))

    
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


def save_best_model(best_guess_filename,model_run_dir,backend):
    """Saves the best model parameters found during the MCMC run."""

    # get samples and log probabilities from backend
    samples = backend.get_chain(flat=True)
    log_prob = backend.get_log_prob(flat=True)

    # find best sample in current samples
    best_idx = np.argmax(log_prob)
    best_logp = log_prob[best_idx]
    best_params = samples[best_idx].tolist()  # convert to list for JSON

    # load an existing best guess, if it exists
    if os.path.exists(best_guess_filename):
        with open(best_guess_filename, "r") as f:
            saved = json.load(f)
        saved_logp = saved["log_prob"]
        saved_params = saved["params"]
    else:
        saved_logp = -np.inf
        saved_params = None

    # save the very best parameters from this run in the run output directory
    with open(model_run_dir + '/best_fit.json', "w") as f:
        json.dump({"log_prob": best_logp, "params": best_params}, f, indent=2)

    # compare and update if better in the all-time best guess file
    if best_logp > saved_logp:
        with open(best_guess_filename, "w") as f:
            json.dump({"log_prob": best_logp, "params": best_params}, f, indent=2)
        print("New best parameters saved.")
    else:
        print("Existing best parameters are better. No update made.")

    # save the rng metadata for the best run from each rank
    for rng_metadata_file in os.listdir(model_run_dir+"/rank_metadata"):
        with open(model_run_dir+"/rank_metadata/"+rng_metadata_file,'r') as f:
            rng_metadata = json.load(f)
        logP = rng_metadata["logProb"]

        # this specifically saves the very best run's metadata into the run output directory
        if logP == best_logp:
            with open(model_run_dir+"/rng_metadata.json",'w') as f:
                json.dump(rng_metadata, f)
            print("yay! found the metadata for the best run!")
            break

            


def run_emcee(model_id,runprops,pool,model_run_dir,dr_path="../data/q1_q17_dr25.csv",expanded_dr_path="../data/expanded_dr25_singles.csv",hsu_star_path="../data/hsu_stellar_catalog_output.csv"):
    """Configures and runs the emcee MCMC sampler."""

    # timer(runprops["timer"],"other readin")

    # define the best guess filename based on model ID
    best_guess_filename = runprops["best_guess_filename"] + f'_{model_id}.json'
    # determine the initial guess filename based on the method. Possible to add a manual filename later.
    initial_guess_filename = best_guess_filename if runprops["initial_guess_method"] == "previous_best" else ""
    # get initial guess positions for the walkers
    p0 = get_initial_guess(runprops["nwalkers"],runprops["ndim"],model_id,method=runprops["initial_guess_method"],previous_filename=initial_guess_filename)
    # assert p0.dtype == np.float32, "params should be a float32"

    # create the emcee backend
    backend_folder = model_run_dir
    os.makedirs(backend_folder, exist_ok=True)
    backend_filename = backend_folder + "/model_" + str(model_id) +".h5"
    if os.path.exists(backend_filename):
        os.remove(backend_filename)
    backend = emcee.backends.HDFBackend(backend_filename)
    backend.reset(runprops["nwalkers"], runprops["ndim"])

    timer(runprops["timer"],"backend setup")

    #### CHECK ABOUT STEP SIZE AND ACCEPTANCE FRACTION...SEEMS LIKE A/FRAC IS VERY LOW, POSSIBLE STOCHAISTICITY ISSUE?
    # create the emcee sampler
    sampler = emcee.EnsembleSampler(runprops["nwalkers"], runprops["ndim"], 
                                    kg_likelihood.parametric_log_probability,backend=backend, pool=pool,moves=[(emcee.moves.StretchMove(a=0.05),1.0)], args=())

    timer(runprops["timer"],"emcee setup")


    if runprops["verbose"]: print("initial guess shape: ", p0.shape)
    assert p0.shape == (runprops["nwalkers"], runprops["ndim"])
    if runprops["verbose"]: print('sampler created. Beginning run.')

    # run mcmc with possibility to thin the chain if desired.
    if runprops['thin_run']:
        state = sampler.run_mcmc(p0, runprops['nburnin']+runprops["nsteps"], progress = True, progress_kwargs={'file':sys.stdout},store = True, thin=runprops["nthinning"])
    else:
        state = sampler.run_mcmc(p0, runprops['nburnin']+runprops["nsteps"], progress = True, progress_kwargs={'file':sys.stdout}, store = True)

    timer(runprops["timer"],"emcee run")

    # save the best model parameters found during this run
    save_best_model(best_guess_filename,model_run_dir,backend)



def main(model_id, runprops):
    """Main function to set up and run the MCMC sampling."""

    voxel_grid = None
    stellar_df = None
    model_run_dir = None
    density_prior_mask = None
    synthetic_multiplier = None
    stellar_info = None

    # rank 0 reads in the voxel grid and stellar dataframe, then broadcasts to all ranks
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        if runprops["verbose"]: print("[Rank 0] reading csv and voxel grid")
        # read in the voxel grid json object and the column names for its dataframes
        with open(runprops["voxel_json_filename"], "r") as f:
            voxel_grid = json.load(f,object_hook=grid_object_hook)
        if runprops["verbose"]: print("read in json voxel file!")

        ####### this functionality is unneccesary now with the likelihood array, methinks
        # with open('../data/dataframe_column_names.json', "r") as f:
        #     df_columns = json.load(f)
        
        # voxel_grid.assign_column_names(df_columns) # this takes a ton of time, dfs in voxel grid should be handled better
        ######


        if runprops["verbose"]: print("[Rank 0] read in voxel grid, created interpolator")
        

        # print(runprops["processed_stellar_data_filename"])
        # read in the stellar dataframe
        stellar_df = pd.read_csv(runprops["processed_stellar_data_filename"],engine='pyarrow')
        # print(stellar_df.columns)

        # extract the relevant stellar_df info into a np array 
        stellar_info = stellar_df[["Rad","Mass"]].to_numpy()
        stellar_info = np.repeat(stellar_info,runprops["synthetic_multiplier"],axis=0)

        if runprops["verbose"]: print("len(stellar_df) after reading in: ",len(stellar_df))
        if runprops["verbose"]: print("[Rank 0] read in stellar df")
        
        # set up the model run output directory (coded by date/time)
        if runprops["date"] == "today":
            runprops["date"] = datetime.now().date().isoformat()
        if runprops["time"] == "now":
            runprops["time"] = datetime.now().time().isoformat()

        model_run_dir = runprops["model_run_output_folder"] + str(model_id) + f"/{(timestamp_folder:=datetime.now().isoformat(timespec='minutes').replace(':','_'))}"
        os.makedirs(model_run_dir,exist_ok=True)
            
        # plot completeness function grids
        if runprops["plot_completeness"]:
            for ecc in [0,0.1,0.5,0.99]:
                for omega in [0,45,90,135,180,225,270,315,360]:
                    MES_grid_plot(voxel_grid,model_run_dir,ecc_fixed=ecc,omega_fixed=omega)
                    
            
            if runprops["verbose"]: print("Rank 0 made mes grid plot!")
        
        # save the output path so that the plotting script can use this info
        with open("model_run_folder.json", "w") as f:
            print("timestamp folder: ", timestamp_folder)
            json.dump({"model_run_folder":timestamp_folder},f) 
        
        # save run properties to the model run directory
        with open(model_run_dir + "/runprops.json", "w", encoding="utf-8") as f:
            json.dump(runprops, f, indent=2)

        # save the priors being used to a json file in the model run directory
        import kg_priors
        priors = kg_priors.PriorArgs().load_priors().get_priors(model_id)
        with open(model_run_dir + "/priors.json", "w") as f:
            json.dump(priors, f, indent=4)


        assert all(voxel_grid.radius_grid_array == np.asarray(radius_grid_array)), "The read-in voxel grid's radius boundary array is not correct!"
        assert all(voxel_grid.mass_grid_array == np.asarray(mass_grid_array)), "The read-in voxel grid's mass boundary array is not correct!"
        assert all(voxel_grid.period_grid_array == np.asarray(period_grid_array)), "The read-in voxel grid's period boundary array is not correct!"
        assert all(voxel_grid.eccentricity_grid_array == np.asarray(eccentricity_grid_array)), "The read-in voxel grid's eccentricity boundary array is not correct!"
        assert all(voxel_grid.omega_grid_array == np.asarray(omega_grid_array)), "The read-in voxel grid's omega boundary array is not correct!"

        density_prior_mask = voxel_grid.get_density_prior_mask()

        # print("density_prior_mask: ", density_prior_mask)
        # print("density_prior_mask shape: ", density_prior_mask.shape)

        synthetic_multiplier = runprops["synthetic_multiplier"]

    

    # broadcast the voxel grid and stellar dataframe to all ranks
    voxel_grid = comm.bcast(voxel_grid,root=0)
    stellar_info = comm.bcast(stellar_info,root=0)
    model_run_dir = comm.bcast(model_run_dir,root=0)
    density_prior_mask = comm.bcast(density_prior_mask,root=0)
    synthetic_multiplier = comm.bcast(synthetic_multiplier,root=0)
    
    if runprops["verbose"]: print("---BROADCAST HAS BEEN COMPLETED---")
    
    kg_likelihood.voxel_grid = voxel_grid
    kg_likelihood.stellar_info = stellar_info
    kg_likelihood.model_run_dir = model_run_dir
    kg_likelihood.model_id = model_id
    kg_likelihood.density_prior_mask = density_prior_mask
    kg_likelihood.synthetic_multiplier = synthetic_multiplier

    # print("kg_likelihood.stellar_df : ",kg_likelihood.stellar_df )
    # print("len(kg_likelihood.stellar_df) : ",len(kg_likelihood.stellar_df ))


    # set up the MPI pool and run emcee
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        try:        
            run_emcee(model_id,runprops,pool,model_run_dir)
            
            # log a successful run
            with open(model_run_dir + '/' + runprops["log_filename"], "a") as file:
                now = datetime.now().isoformat()
                file.write("success: Model id "+str(model_id) + " " + now + "\n")

            sys.exit(0)
        except Exception as e:
            print("Error occurred..." + str(e))
            with open(model_run_dir + '/' + runprops["log_filename"], "a") as file:
                file.write(str(e)+" Model id: "+str(model_id)+"\n")
                file.write(f"errored at {datetime.now().isoformat()}!")
        finally:
            timer(runprops["timer"],"",mode="final")
    

if __name__ == "__main__":

    # for timing purposes
    old_time = time.time()
    start_time = old_time

    # needs to specify which model is being run (so far, only 0 is supported)
    if len(sys.argv) != 2:
        print("invalid input. Enter which mixture model you want to run.")
        sys.exit(1)
    model_id = int(sys.argv[1])
    
    # Verify the correct path script is being run from. 
    cwd = os.getcwd()
    print(cwd)        
    if 'src' in cwd:
        runprops_filename = "../runs/param_runprops.txt"
    elif 'runs' in cwd:
        runprops_filename = "param_runprops.txt"
    elif 'results' in cwd:
        runprops_filename = "param_runprops.txt"
    else:
        print('you are not starting from a proper directory. you should run kg_run_param.py from a src, runs, or a results directory.')
        sys.exit(1)
    
    # Get runprops loaded in, find the initial guess file.
    getData = ReadJson(runprops_filename)
    runprops = getData.outProps()

    # run the main script
    main(model_id,runprops)