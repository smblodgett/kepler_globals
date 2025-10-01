# main script that runs Kepler Globals modeling: as of now, this is modeling the voxels of the M-P-R relationship,
# assuming each voxel's independence

# begun October 3, 2024
# developed by Steven Blodgett, with Darin Ragozzine, Dallin Spencer, and Daniel Jones
# codebase drawn from Dallin Spencer's multi_moon

from mpi4py import MPI
import os

# rank initialization signature
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
print(f"[Rank {rank}/{size}] starting up")
print(os.system("hostname"))
# space out the walkers by a tenth of a second
import time
time.sleep(.05*rank) 

import pandas as pd
import numpy as np
import sys
import emcee
from datetime import datetime
import json
from schwimmbad import MPIPool

import kg_likelihood
from kg_griddefiner import *
from kg_param_initial_guess import get_initial_guess
from kg_utilities import ReadJson
from kg_plots import MES_grid_plot

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
    

# def number_of_singles_in_voxel(voxel, expanded_dr_df):
#     """Returns the number of singles in a given voxel."""
#     mask = ((expanded_dr_df["radius"] < voxel.top_radius) &
#             (expanded_dr_df["radius"] > voxel.bottom_radius) &
#             (expanded_dr_df["period"] < voxel.top_period) &
#             (expanded_dr_df["period"] > voxel.bottom_period) &
#             (expanded_dr_df["mass"] < voxel.top_mass) &
#             (expanded_dr_df["mass"] > voxel.bottom_mass)
#             )
    
#     print("singles length: ",len(expanded_dr_df.loc[mask]))
#     return len(expanded_dr_df.loc[mask])

def save_best_model(best_guess_filename,backend):
    samples = backend.get_chain(flat=True)
    log_prob = backend.get_log_prob(flat=True)

    # === FIND BEST SAMPLE IN CURRENT BACKEND ===
    best_idx = np.argmax(log_prob)
    best_logp = log_prob[best_idx]
    best_params = samples[best_idx].tolist()  # convert to list for JSON

    # === LOAD EXISTING BEST GUESS, IF IT EXISTS ===
    if os.path.exists(best_guess_filename):
        with open(best_guess_filename, "r") as f:
            saved = json.load(f)
        saved_logp = saved["log_prob"]
        saved_params = saved["params"]
    else:
        saved_logp = -np.inf
        saved_params = None

    # === COMPARE AND UPDATE IF BETTER ===
    if best_logp > saved_logp:
        with open(best_guess_filename, "w") as f:
            json.dump({"log_prob": best_logp, "params": best_params}, f, indent=2)
        print("New best parameters saved.")
    else:
        print("Existing best parameters are better. No update made.")


def run_emcee(model_id,runprops,pool,model_run_dir,dr_path="../data/q1_q17_dr25.csv",expanded_dr_path="../data/expanded_dr25_singles.csv",hsu_star_path="../data/hsu_stellar_catalog_output.csv"):
    """Configures and runs the emcee MCMC sampler.""" # DON"T FORGET POOL

    # # Get DR25 catalog.
    # dr_df = pd.read_csv(dr_path)
    # # Get processed singles dataframe. 
    # expanded_dr_singles_df = pd.read_csv(expanded_dr_path) # These have already been filtered for the Hsu stellar catalog (and given the proper values)
    # # Get Hsu stellar catalog.
    # hsu_star_df = pd.read_csv(hsu_star_path)
    # # Mark the DR25 catalog with those that are in the Hsu stellar catalog.
    # dr_df["in_hsu"] = dr_df["kepid"].isin(hsu_star_df["kepid"]).astype(int)
    
    # timer(runprops["timer"],"other readin")

    best_guess_filename = runprops["best_guess_filename"] + f'_{model_id}.json'

    initial_guess_filename = best_guess_filename if runprops["initial_guess_method"] == "previous_best" else ""

    p0 = get_initial_guess(runprops["nwalkers"],runprops["ndim"],model_id,method=runprops["initial_guess_method"],previous_filename=initial_guess_filename)


    # Create the emcee backend.
    backend_folder = model_run_dir
    os.makedirs(backend_folder, exist_ok=True)
    backend_filename = backend_folder + "/model_" + str(model_id) +".h5"
    if os.path.exists(backend_filename):
        os.remove(backend_filename)
    backend = emcee.backends.HDFBackend(backend_filename)
    backend.reset(runprops["nwalkers"], runprops["ndim"])

    timer(runprops["timer"],"backend setup")

    
    print("type(pool): ",type(pool))
    print("pool: ",pool)

    # Create the emcee sampler.
    sampler = emcee.EnsembleSampler(runprops["nwalkers"], runprops["ndim"], 
                                    kg_likelihood.parametric_log_probability,backend=backend, pool=pool, args=())

    timer(runprops["timer"],"emcee setup")


    print("initial guess shape: ", p0.shape)
    assert p0.shape == (runprops["nwalkers"], runprops["ndim"])

    if runprops["verbose"]: print('sampler created. Beginning run.')

    if runprops['thin_run']:
        state = sampler.run_mcmc(p0, runprops['nburnin']+runprops["nsteps"], progress = True, progress_kwargs={'file':sys.stdout},store = True, thin=runprops["nthinning"])
    else:
        state = sampler.run_mcmc(p0, runprops['nburnin']+runprops["nsteps"], progress = True, progress_kwargs={'file':sys.stdout}, store = True)

    timer(runprops["timer"],"emcee run")

    save_best_model(best_guess_filename,backend)



def main(model_id, runprops):  ## don't forget pool!

    def grid_object_hook(dct):
        # 1) RPMeoVoxel: has eccentricity & omega bounds
        keys = set(dct)
        if {
            "bottom_radius","top_radius",
            "bottom_period","top_period",
            "bottom_mass","top_mass",
            "bottom_eccentricity","top_eccentricity",
            "bottom_omega","top_omega"
        }.issubset(keys):
            v = RPMeoVoxel(
                dct["bottom_radius"], dct["top_radius"],
                dct["bottom_period"], dct["top_period"],
                dct["bottom_mass"],  dct["top_mass"],
                dct["bottom_eccentricity"], dct["top_eccentricity"],
                dct["bottom_omega"], dct["top_omega"],
            )
            v.id_number = dct.get("id_number", -1)
            if "df" in dct:
                v.df = pd.DataFrame(dct["df"])
                v.is_add_data = True
            return v

        # 2) RPMeoGrid: must have all five edge arrays + voxel_array + the two prob arrays + id_array
        if {
            "radius_grid_array","period_grid_array","mass_grid_array",
            "eccentricity_grid_array","omega_grid_array",
            "voxel_array","p_detection_array","p_transit_array","id_array"
        }.issubset(keys):

            # 2a) Reconstruct the grid object
            grid = RPMeoGrid(
                dct["radius_grid_array"],
                dct["period_grid_array"],
                dct["mass_grid_array"],
                dct["eccentricity_grid_array"],
                dct["omega_grid_array"],
            )
            # 2b) Overwrite its raw arrays
            grid.voxel_array         = np.array(dct["voxel_array"],    dtype=object)
            grid.p_detection_array   = np.array(dct["p_detection_array"])
            grid.p_transit_array     = np.array(dct["p_transit_array"])
            grid.id_array            = np.array(dct["id_array"])
            grid.likelihood_array    = np.array(dct["likelihood_array"])

            # 2c) Rebuild the interpolators so .p_detection_interp exists
            grid.p_detection_interp = RegularGridInterpolator(
                (grid.radius_grid_array,
                grid.period_grid_array,
                grid.mass_grid_array,
                grid.eccentricity_grid_array,
                grid.omega_grid_array),
                grid.p_detection_array
            )
            grid.p_transit_interp = RegularGridInterpolator(
                (grid.radius_grid_array,
                grid.period_grid_array,
                grid.mass_grid_array,
                grid.eccentricity_grid_array,
                grid.omega_grid_array),
                grid.p_transit_array
            )

            return grid

        # fallback: leave dict alone
        return dct

    # print(type(voxel_grid))
    # # time.sleep(5)
    # print(voxel_grid)
    # # time.sleep(10)

    # for voxel in voxel_grid.voxel_array.flat:
    #     print(voxel)
    voxel_grid = None
    stellar_df = None

    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        print("[Rank 0] reading csv and voxel grid")
        with open(runprops["voxel_json_filename"], "r") as f:
            voxel_grid = json.load(f,object_hook=grid_object_hook)
        with open('../data/dataframe_column_names.json', "r") as f:
            df_columns = json.load(f)
        voxel_grid.assign_column_names(df_columns)
        MES_grid_plot(voxel_grid.p_detection_interp,voxel_grid.p_transit_interp,runprops["completeness_plot_folder"])
        print("[Rank 0 made mes grid plot!")
        stellar_df = pd.read_csv(runprops["processed_stellar_data_filename"])
        print("[Rank 0 read in stellar df]")
        if runprops["date"] == "today":
            runprops["date"] = datetime.now().date().isoformat()
        if runprops["time"] == "now":
            runprops["time"] = datetime.now().time().isoformat()

        model_run_dir = runprops["model_run_output_folder"] + str(model_id) + f"/{(timestamp_folder:=datetime.now().isoformat(timespec='minutes').replace(':','_'))}"
        os.makedirs(model_run_dir,exist_ok=True)

        with open(model_run_dir + "/runprops.json", "w", encoding="utf-8") as f:
            json.dump(runprops, f, indent=2)

        import kg_priors
        prior_args_json = {k: list(v) for k,v in kg_priors.prior_args.items()}

        with open(model_run_dir + "/priors.json", "w") as f:
            json.dump(prior_args_json, f, indent=4)
    

    voxel_grid = comm.bcast(voxel_grid,root=0)
    stellar_df = comm.bcast(stellar_df,root=0)
    
    print("---BROADCAST HAS BEEN COMPLETED---")
    
    kg_likelihood.voxel_grid = voxel_grid
    kg_likelihood.stellar_df = stellar_df


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
            
            with open("model_run_folder.json", "a") as f:
                json.dump({"model_run_folder":timestamp_folder},f) # so that the plotting script can use this

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