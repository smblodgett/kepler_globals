from mpi4py import MPI
import os

# rank initialization signature
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
print(f"[Rank {rank}/{size}] starting up")
print(os.system("hostname"))
# space out the walkers by a tenth of a second
import time
time.sleep(.02*rank) 


import os
import sys
import math
import numbers
import numpy as np
import pandas as pd
import json
from kg_utilities import ReadJson, mass_given_density_radius
from kg_griddefiner import RPMeoGrid, RPMeoVoxel
from kg_param_boundary_arrays import radius_grid_array, period_grid_array, mass_grid_array, eccentricity_grid_array, omega_grid_array
from kg_constants import G, RECM




class GridJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, numbers.Real):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj

        if isinstance(obj, pd.DataFrame):
            return obj.where(pd.notnull(obj), None).to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.where(pd.notnull(obj), None).to_dict()
        elif isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.number):
                return np.where(np.isnan(obj), None, obj).tolist()
            else:
                return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return {k: self.default(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.default(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self.default(i) for i in obj)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        return str(obj)


def sample_eccentricity_omega(radius, period, b, T_14,rho_star_true):
    """
    Samples eccentricity and omega for a planet based on its radius and period, using the photoeccentric effect.
    
    See MacDougal, Gilbert, and Pettigura 2023. 
    """
    i=1000
    eccentricity = np.random.uniform(0, 0.9,size=i)  # Sample eccentricity uniformly between 0 and 0.9
    omega = np.random.uniform(0, 360,size=i)  # Sample omega uniformly between 0 and 360 degrees
    rho_star_sample = (3 * np.pi / ((G * 1000 * 100**3) * (period * 24 * 3600)**2)) * ((((1+radius*RECM)**2) - b**2)/ (np.sin((T_14 * np.pi / period) * ( (1+ eccentricity*np.sin(omega * np.pi / 180))/ np.sqrt(1-eccentricity**2)))**2) + b**2)**1.5  # Sample stellar density based on the photoeccentric effect
    log_likelihood = -0.5 * (rho_star_sample - rho_star_true)**2

    return eccentricity, omega


def process_singles_df(singles_dr_df,stellar_df):

    final_singles_array = np.zeros((len(singles_dr_df)*1000,5)) # radius, period, mass, eccentricity, omega

    for index, row in singles_dr_df.iterrows():
        radius = np.random.normal(row["koi_prad"], np.maximum(np.abs(row["koi_prad_err1"]), np.abs(row["koi_prad_err2"])),size=1000)
        period = np.random.normal(row["koi_period"], np.maximum(np.abs(row["koi_period_err1"]), np.abs(row["koi_period_err2"])),size=1000)
        b = np.random.normal(row["koi_impact"], np.maximum(np.abs(row["koi_impact_err1"]), np.abs(row["koi_impact_err2"])),size=1000)
        T_14 = np.random.normal(row["koi_duration"], np.maximum(np.abs(row["koi_duration_err1"]), np.abs(row["koi_duration_err2"])),size=1000)

        density = np.random.uniform(0.01, 10, size=1000) 
        mass = mass_given_density_radius(density, radius)
        rho_star_true = stellar_df[stellar_df["KIC"]==row["kepid"]]["dens"].values[0]
        eccentricity, omega = sample_eccentricity_omega(radius, period, b, T_14,rho_star_true)

        final_singles_array[index*1000:(index+1)*1000] = [radius, period, mass, eccentricity, omega]

    return pd.DataFrame(final_singles_array, columns=["radius","period","mass","eccentricity","omega"])


def main(runprops):
    
    use_cache = os.path.isdir(runprops["voxel_data_folder"]) and not runprops["reload_KMDC"]


    voxel_grid = None
    stellar_df = None
    stellar_df_reduced = None
    comm = MPI.COMM_WORLD
    # with MPIPool() as pool:
        # if not pool.is_master():
        #     pool.wait()
        #     sys.exit(0)

    if comm.Get_rank() == 0:

        if not runprops["suppress_warnings"]: 
            if not use_cache:
                print("Warning! use_cache is",use_cache,"meaning that this run will take a long time!")
                print("Only run this way if your voxel data hasn't yet been cached.")
        
        # If the voxels don't have their data cached, then read in everything.
        if not use_cache:
            df = pd.read_csv(runprops["input_data_filename"],index_col=0,engine='pyarrow')
            if runprops["verbose"]: print("read in the catalog without caching (press enter to continue)")
            # input()
            print("now we're caching it!")
            df = df[["R_pE","Period_days","M_pE","e","omega","KIC","rho_p","planet"]]#,"p_trans","MES_rowe"]]
            #df = create_probability_weighted(df)
            df.to_csv(runprops["input_data_folder"]+"/KMDC_RPMeo.csv")
            if runprops["verbose"]: print("data has been cached for future runs!")
            # input()
        # Otherwise, you can just read in 1 voxel that has its data cached.    
        else:
            df = pd.read_csv(runprops["input_data_folder"]+"/KMDC_RPMeo.csv",index_col=0,engine='pyarrow')
            if runprops["verbose"]: print("read in cached df")

        print("full data df: ",df)

        # Get DR25 catalog for the inclusion of singles in the catalog.
        dr_df = pd.read_csv(runprops['dr_25_data_filename'],engine='pyarrow')

        dr_df['multiplicity'] = dr_df['kepid'].map(dr_df['kepid'].value_counts())

        # dr_df = dr_df[dr_df["koi_disposition"]=="CONFIRMED"] #??
        singles_dr_df = dr_df[dr_df["multiplicity"]==1]  #?? this is the cut for singles, but we should also make sure to exclude any of these that are in the df already as multis.

        

        # Setup and load grid with data. If data is not cached, then cache data from whole grid into voxel dataframes.
        voxel_grid = RPMeoGrid(radius_grid_array, period_grid_array, mass_grid_array, eccentricity_grid_array, omega_grid_array)

        voxel_grid.setup_dataframes(df.columns)


        print("initialized voxel grid!")

        gaia_df = pd.read_csv(runprops["gaia_data_filename"],delimiter='\t',header=1,engine='pyarrow')
        gaia_df = gaia_df[["KIC","Mass","Teff","Rad"]]

        print("starting stellar df")

        stellar_df = pd.read_csv(runprops["stellar_data_filename"],engine='pyarrow')
        stellar_df = stellar_df[stellar_df["st_delivname"]=="q1_q17_dr25_stellar"]
        stellar_df = stellar_df.rename(columns={"kepid":"KIC"})


        # stellar_df = stellar_df.merge(gaia_df, on='KIC', how='left')

        # for old_col,new_col in zip(["teff","mass","radius"],["Teff","Mass","Rad"]):
        #     stellar_df[old_col] = stellar_df[new_col].combine_first(stellar_df[old_col])


        stellar_df = stellar_df[(stellar_df["teff"]>4000) & (stellar_df["teff"]<7000)]
        stellar_df = stellar_df[(stellar_df["logg"]>4)]

        stellar_df = stellar_df[(~stellar_df["mass"].isna()) & (~stellar_df["limbdark_coeff1"].isna()) & (~stellar_df["teff"].isna())]
        
        print("stellar df cuts applied")

        stellar_df_kics = stellar_df["KIC"]

        df = df[df["KIC"].isin(stellar_df_kics)]

        singles_dr_df = singles_dr_df[singles_dr_df["kepid"].isin(stellar_df_kics)]

        processed_singles_dr_df = process_singles_df(singles_dr_df,stellar_df)

        if runprops["exclude_high_densities"]:
            df = df[df["rho_p"]<runprops["maximum_density"]]


        df['unique_planet'] = df['KIC'] + df['planet']
        kic_dict = df['unique_planet'].value_counts().to_dict()

        print("kic_dict: ", kic_dict)

        voxel_grid.set_kic_dict(kic_dict)
        

        print("length of df after matching to stellar catalog, filtering densities: ",len(df))

        voxel_grid.add_data(df)

        stellar_df_reduced=stellar_df.sample(n=100,random_state=22)

    # reweight the remaining planets in the df to account for removing unrealistic densities


    # sampling ecc and omega for the singles, and adding them cool - Gilbert and Pettigura look it up 
    # photoeccentric effect
    # Planets larger than Neptune have elevated eccentricies
    # cites the accurate and whatever photoeccentric sampling

    voxel_grid = comm.bcast(voxel_grid,root=0)
    stellar_df = comm.bcast(stellar_df,root=0)
    stellar_df_reduced = comm.bcast(stellar_df_reduced,root=0)

    print("broadcasted voxel grid and stellar df")
        
    voxel_grid.setup_completeness_grid(stellar_df_reduced,comm) # this is the kepler stellar catalog, which has the stellar radii and masses
    print("set up completeness grid")
    voxel_grid.setup_likelihood_grid()
    # MES_grid_plot(voxel_grid.p_detection_interp,voxel_grid.p_transit_interp,runprops["completeness_plot_folder"])
    
    if runprops["verbose"] and comm.Get_rank() == 0: print("MES grid has been set up!")

    comm.Barrier()

    if comm.Get_rank() == 0:
        grid_string = json.dumps(voxel_grid,cls=GridJSONEncoder)

        
        with open(runprops["voxel_json_filename"], "w") as f:
            f.write(grid_string)

        stellar_df.to_csv("../data/keplerstellar_with_cuts.csv")


        df_columns = json.dumps(list(df.columns))
        with open('../data/dataframe_column_names.json', "w") as f:
            f.write(df_columns)
        
        print("Finished writing to json!")
    
    comm.Barrier()



if __name__ == "__main__":       
    
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

    main(runprops)