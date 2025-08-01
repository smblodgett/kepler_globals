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
from datetime import datetime
import json
from schwimmbad import MPIPool
from mpi4py import MPI

import kg_likelihood
from kg_griddefiner import *
from kg_param_boundary_arrays import radius_grid_array, period_grid_array, mass_grid_array, eccentricity_grid_array, omega_grid_array
from kg_param_initial_guess import get_initial_guess
from kg_utilities import ReadJson, create_probability_weighted
from kg_probability_distributions import get_MES
from kg_plots import MES_grid_plot

    
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


def run_emcee(model_id,runprops,pool,dr_path="../data/q1_q17_dr25.csv",expanded_dr_path="../data/expanded_dr25_singles.csv",hsu_star_path="../data/hsu_stellar_catalog_output.csv"):
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
    backend_folder = runprops["results_folder"] + "/param_backend/"
    os.makedirs(backend_folder, exist_ok=True)
    backend_filename = backend_folder + "param_model_" + str(model_id) + ".h5"
    if os.path.exists(backend_filename):
        os.remove(backend_filename)
    backend = emcee.backends.HDFBackend(backend_filename)
    backend.reset(runprops["nwalkers"], runprops["ndim"])

    timer(runprops["timer"],"backend setup")

    
    print("type(pool): ",type(pool))
    print("pool: ",pool)

    # Create the emcee sampler.
    sampler = emcee.EnsembleSampler(runprops["nwalkers"], runprops["ndim"], 
                                    kg_likelihood.parametric_log_probability,backend=backend, pool=pool, args=())#,pool=pool)

    timer(runprops["timer"],"emcee setup")


    print("initial guess shape: ", p0.shape)
    assert p0.shape == (runprops["nwalkers"], runprops["ndim"])

    if runprops["verbose"]: print('sampler created. Beginning run.')

    if runprops['thin_run']:
        state = sampler.run_mcmc(p0, runprops['nburnin']+runprops["nsteps"], progress = True, store = True, thin=runprops["nthinning"])
    else:
        state = sampler.run_mcmc(p0, runprops['nburnin']+runprops["nsteps"], progress = True, store = True)

    timer(runprops["timer"],"emcee run")

    save_best_model(best_guess_filename,backend)



def main(model_id, runprops):  ## don't forget pool!

    # use_cache = os.path.isdir(runprops["voxel_data_folder"]) and not runprops["reload_KMDC"]

    # if not runprops["suppress_warnings"]: 
    #     if not use_cache:
    #         print("Warning! use_cache is",use_cache,"meaning that this run will take a long time!")
    #         print("Only run this way if your voxel data hasn't yet been cached.")
    
    # # If the voxels don't have their data cached, then read in everything.
    # if not use_cache:
    #     df = pd.read_csv(runprops["input_data_filename"],index_col=0,engine='pyarrow')
    #     if runprops["verbose"]: print("read in the catalog without caching (press enter to continue)")
    #     # input()
    #     print("now we're caching it!")
    #     df = df[["R_pE","Period_days","M_pE","e","omega"]]#,"p_trans","MES_rowe"]]
    #     #df = create_probability_weighted(df)
    #     df.to_csv(runprops["input_data_folder"]+"/KMDC_RPMeo.csv")
    #     if runprops["verbose"]: print("data has been cached for future runs! (press enter to continue)")
    #     # input()
    # # Otherwise, you can just read in 1 voxel that has its data cached.    
    # else:
    #     df = pd.read_csv(runprops["input_data_folder"]+"/KMDC_RPMeo.csv",index_col=0,engine='pyarrow')
    #     if runprops["verbose"]: print("read in cached df")

    # print("full data df: ",df)
    
    # timer(runprops["timer"],"data readin")

    # # Setup and load grid with data. If data is not cached, then cache data from whole grid into voxel dataframes.
    # voxel_grid = RPMeoGrid(radius_grid_array, period_grid_array, mass_grid_array, eccentricity_grid_array, omega_grid_array)
    # voxel_grid.setup_dataframes(df.columns)
    # voxel_grid.add_data(df)

    # gaia_df = pd.read_csv(runprops["gaia_data_filename"],delimiter='\t',header=1,engine='pyarrow')
    # gaia_df = gaia_df[["KIC","Mass","Teff","Rad"]]

    # stellar_df = pd.read_csv(runprops["stellar_data_filename"],engine='pyarrow')
    # stellar_df = stellar_df[stellar_df["st_delivname"]=="q1_q17_dr25_stellar"]
    # stellar_df = stellar_df.rename(columns={"kepid":"KIC"})


    # stellar_df = stellar_df.merge(gaia_df, on='KIC', how='left')

    # for old_col,new_col in zip(["teff","mass","radius"],["Teff","Mass","Rad"]):
    #     stellar_df[old_col] = stellar_df[new_col].combine_first(stellar_df[old_col])


    # stellar_df = stellar_df[(stellar_df["teff"]>4000) & (stellar_df["teff"]<7000)]
    # stellar_df = stellar_df[(stellar_df["logg"]>4)]

    # stellar_df = stellar_df[(~stellar_df["mass"].isna()) & (~stellar_df["limbdark_coeff1"].isna()) & (~stellar_df["teff"].isna())]
    # voxel_grid.setup_completeness_grid(stellar_df) # this is the kepler stellar catalog, which has the stellar radii and masses
    # MES_grid_plot(voxel_grid.p_detection_interp,voxel_grid.p_transit_interp,runprops["completeness_plot_folder"])
    # if runprops["verbose"]: print("MES grid has been set up!")

    # timer(runprops["timer"],"weight partition")

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


    with open(runprops["voxel_json_filename"], "r") as f:
        voxel_grid = json.load(f,object_hook=grid_object_hook)
    
    
    with open('../data/dataframe_column_names.json', "r") as f:
        df_columns = json.load(f)

    voxel_grid.assign_column_names(df_columns)

    # print(type(voxel_grid))
    # # time.sleep(5)
    # print(voxel_grid)
    # # time.sleep(10)

    # for voxel in voxel_grid.voxel_array.flat:
    #     print(voxel)
    
    kg_likelihood.voxel_grid = voxel_grid

    kg_likelihood.stellar_df = pd.read_csv(runprops["processed_stellar_data_filename"])

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        try:        
            run_emcee(model_id,runprops,pool)
            sys.exit(0)
        except Exception as e:
            print("Error occurred..." + str(e))
            with open(runprops["log_filename"], "a") as file:
                file.write(str(e)+" Model: "+str(model_id)+"\n")
        finally:
            timer(runprops["timer"],"",mode="final")
    


if __name__ == "__main__":

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    print(f"[Rank {rank}/{size}] starting up")

    old_time = time.time()
    start_time = old_time
    if len(sys.argv) != 2:
        print("invalid input. Enter which mixture model you want to run.")
        sys.exit(1)
        
    model_id = int(sys.argv[1])
    
    # Verify the correct path script is being run from. 
    cwd = os.getcwd()
    print(cwd)        

    # Find the runprops file path. 
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
    main(model_id,runprops)

    with open(runprops["log_filename"], "a") as file:
        now = datetime.now().isoformat()

        file.write("success: Model "+str(model_id)+ now + "\n")
    

# figure out a way to visualize this...maybe show it before setup? then again after doing emcee? figure 

# run MCMC on each cell in the voxel grid:
# any voxel that is outside of the density prior should not even be run through emcee...
# if we have an empty voxel, then skip and output a list of voxels, flagging empty ones based on why they're empty
# for the impossibly heavy planets, we could absolutely put a prior for density