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
from kg_constants import G, RECM, RHOS, RSCM, MSKG, MEKG
from kg_plots import ecc_omega_singles_posterior_plot




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


def sample_eccentricity_omega(planet_star_radius_ratio, period, b, T_14,rho_star_true, rho_star_uncertainty,KIC_id,num_samples):
    """
    Samples eccentricity and omega for a planet based on its radius and period, using the photoeccentric effect.
    
    See MacDougal, Gilbert, and Pettigura 2023. 
    """
    i=num_samples
    eccentricity = np.random.uniform(0, 0.99,size=i)  # Sample eccentricity uniformly between 0 and 0.99
    omega = np.random.uniform(0, 360,size=i)  # Sample omega uniformly between 0 and 360 degrees


    rho_star_sample = np.zeros(i)  # Initialize an array to store the sampled stellar densities

    inside = (((1+planet_star_radius_ratio)**2) - b**2)/ (np.sin(((T_14 / 24) * np.pi / period) * ( (1+ eccentricity*np.sin(omega * np.pi / 180))/ np.sqrt(1-eccentricity**2)))**2) + b**2
    
    valid = inside > 0

    rho_star_sample[valid] = (3 * np.pi / (G * (period[valid] * 24 * 3600)**2)) * (inside[valid])**1.5  # Sample stellar density based on the photoeccentric effect
    
    
    log_likelihood = np.full(i, -np.inf)  # Initialize log-likelihood array with negative infinity

    print("rho_star_sample: ",rho_star_sample)
    
    log_likelihood[valid] = -0.5 * (rho_star_sample[valid] - rho_star_true)**2 / rho_star_uncertainty**2  # Calculate log-likelihood based on the difference between the sampled and true stellar density

    print("log_likelihood: ",log_likelihood)

    weight = np.exp(log_likelihood) / np.sum(np.exp(log_likelihood))  # Normalize the weights

    print("weight :",weight)

    indices = np.random.choice(range(i), size=i, p=weight)  # Sample indices based on the weights

    eccentricity = eccentricity[indices]  # Sample eccentricity based on the weights
    omega = omega[indices]  # Sample omega based on the weights

    ecc_omega_singles_posterior_plot(eccentricity,omega,KIC_id=KIC_id)

    ### how to get the eccentricity to be less elevated? How do we deweight the high eccentricities?
    ### cut out all samples with q < 2 stellar radii and reweight?

    return eccentricity, omega


def process_singles_df(singles_dr_df,stellar_df):

    num_posteriors_per_planet = 1000

    final_singles_array = np.zeros((len(singles_dr_df)*num_posteriors_per_planet,6)) # radius, period, mass, eccentricity, omega

    ##### graphing GJ436 for validation - using Lanotte et al 2014
    radius = np.random.normal(3.96,0.05,size=num_posteriors_per_planet)
    period = np.random.normal(2.6438979,0.0000003,size=num_posteriors_per_planet)
    b = np.random.normal(0.8521,0.0021,size=num_posteriors_per_planet)
    T_14 = np.random.normal(0.04227*24,0.00016*24,size=num_posteriors_per_planet)
    rho_star_true = (0.452 * MSKG * 1000) / ((4/3) * np.pi * (0.455 * RSCM)**3) * 1000
    rho_star_uncertainty_lower = ((0.452 - 0.012) * MSKG * 1000) / ((4/3) * np.pi * ((0.455 + 0.014) * RSCM)**3) * 1000
    rho_star_uncertainty_upper = ((0.452 + 0.014) * MSKG * 1000) / ((4/3) * np.pi * ((0.455 - 0.012) * RSCM)**3) * 1000
    rho_star_uncertainty = np.maximum(np.abs(rho_star_uncertainty_lower - rho_star_true), np.abs(rho_star_uncertainty_upper - rho_star_true))
    star_planet_radius_ratio = radius * RECM / (0.455 * RSCM)
    print("GJ rho star true: ",rho_star_true)
    print("GJ rho star uncertainty: ",rho_star_uncertainty)
    sample_eccentricity_omega(star_planet_radius_ratio, period, b, T_14,rho_star_true,rho_star_uncertainty,"GJ436",num_posteriors_per_planet)
    #####

    for index, row in singles_dr_df.iterrows():
        radius = np.random.normal(row["koi_prad"], np.maximum(np.abs(row["koi_prad_err1"]), np.abs(row["koi_prad_err2"])),size=num_posteriors_per_planet)
        period = np.random.normal(row["koi_period"], np.maximum(np.abs(row["koi_period_err1"]), np.abs(row["koi_period_err2"])),size=num_posteriors_per_planet)
        print("period with max abs error:", row["koi_period"], np.maximum(np.abs(row["koi_period_err1"]), np.abs(row["koi_period_err2"])))

        b = np.random.normal(row["koi_impact"], np.maximum(np.abs(row["koi_impact_err1"]), np.abs(row["koi_impact_err2"])),size=num_posteriors_per_planet)
        T_14 = np.random.normal(row["koi_duration"], np.maximum(np.abs(row["koi_duration_err1"]), np.abs(row["koi_duration_err2"])),size=num_posteriors_per_planet)

        print("radius: ",radius)
        print("number of NaN in radius: ",np.sum(np.isnan(radius)))
        print("period: ",period)
        print("number of NaN in period: ",np.sum(np.isnan(period)))
        print("b: ",b)
        print("number of NaN in b: ",np.sum(np.isnan(b)))
        print("T_14: ",T_14)
        print("number of NaN in T_14: ",np.sum(np.isnan(T_14)))


        density = np.random.uniform(0.01, 10, size=num_posteriors_per_planet) 
        mass = mass_given_density_radius(density, radius)

        print("mass: ",mass)
        print("number of NaN in mass: ",np.sum(np.isnan(mass)))

        # make sure the units here are right, the log uncertainties are weird. 

        rho_star_true_log = stellar_df[stellar_df["KIC"]==row["kepid"]]["rho"].values[0]
        rho_star_true = 10**(rho_star_true_log) * RHOS
        rho_star_upper_uncertainty = stellar_df[stellar_df["KIC"]==row["kepid"]]["E_rho"].values[0]
        rho_star_upper_uncertainty =  10**(rho_star_upper_uncertainty) * RHOS
        rho_star_lower_uncertainty = stellar_df[stellar_df["KIC"]==row["kepid"]]["e_rho"].values[0]
        rho_star_lower_uncertainty = - 10**(rho_star_lower_uncertainty) * RHOS
        rho_star_uncertainty = np.maximum(np.abs(rho_star_upper_uncertainty), np.abs(rho_star_lower_uncertainty))

        radius_star_val = stellar_df[stellar_df["KIC"]==row["kepid"]]["Rad"].values[0]
        radius_star_upper_uncertainty = stellar_df[stellar_df["KIC"]==row["kepid"]]["E_Rad"].values[0]
        radius_star_lower_uncertainty = stellar_df[stellar_df["KIC"]==row["kepid"]]["e_Rad"].values[0]
        radius_star_uncertainty = np.maximum(np.abs(radius_star_upper_uncertainty), np.abs(radius_star_lower_uncertainty))
        radius_star = np.random.normal(radius_star_val, radius_star_uncertainty, size=num_posteriors_per_planet)


        planet_star_radius_ratio = radius * RECM / (radius_star * RSCM) 

        print("rho_star_true: ",rho_star_true)
        print("rho_star_uncertainty: ",rho_star_uncertainty)
        print("number of NaN in rho_star_true: ",np.sum(np.isnan(rho_star_true)))
        print("number of NaN in rho_star_uncertainty: ",np.sum(np.isnan(rho_star_uncertainty)))

        eccentricity, omega = sample_eccentricity_omega(planet_star_radius_ratio, period, b, T_14,rho_star_true,rho_star_uncertainty,row["kepid"],num_posteriors_per_planet)

        final_singles_array[index*num_posteriors_per_planet:(index+1)*num_posteriors_per_planet] = np.array([radius, period, mass, eccentricity, omega,np.full(shape=num_posteriors_per_planet,fill_value=row["kepid"])]).T

    df = pd.DataFrame(final_singles_array, columns=["R_pE","Period_days","M_pE","e","omega","kepid"])

    return df


def main(runprops):
    
    use_cache = os.path.isdir(runprops["voxel_data_folder"]) and not runprops["reload_KMDC"]

    # Define important variables for later in namespace
    voxel_grid = None
    stellar_df = None
    stellar_df_reduced = None
    final_kdc_df = None
    comm = MPI.COMM_WORLD


    if comm.Get_rank() == 0:

        if not runprops["suppress_warnings"]: 
            if not use_cache:
                print("Warning! use_cache is",use_cache,"meaning that this run will take a long time!")
                print("Only run this way if your voxel data hasn't yet been cached.")
        
        # If the voxels don't have their data cached, then read in everything.
        if not use_cache:
            df = pd.read_csv(runprops["input_data_filename"],index_col=0,engine='pyarrow')
            if runprops["verbose"]: print("read in the catalog without caching (press enter to continue)")
            print("now we're caching it!")
            df = df[["R_pE","Period_days","M_pE","e","omega","KIC","rho_p","planet"]]#,"p_trans","MES_rowe"]]
            #df = create_probability_weighted(df)
            df.to_csv(runprops["input_data_folder"]+"/KMDC_RPMeo.csv")
            if runprops["verbose"]: print("data has been cached for future runs!")
            
        # Otherwise, you can just read in 1 voxel that has its data cached.    
        else:
            df = pd.read_csv(runprops["input_data_folder"]+"/KMDC_RPMeo.csv",index_col=0,engine='pyarrow')
            if runprops["verbose"]: print("read in cached df")

        print("full data df: ",df)

        # Get DR25 catalog for the inclusion of singles in the catalog.
        dr_df = pd.read_csv(runprops['dr_25_data_filename'],engine='pyarrow')

        # create a column for the multipliity of each DR25 planet
        dr_df['multiplicity'] = dr_df['kepid'].map(dr_df['kepid'].value_counts())


        # dr_df = dr_df[dr_df["koi_disposition"]=="CONFIRMED"] #??  - do we hit only the "confirmed" planets? I don't think so


        singles_dr_df = dr_df[dr_df["multiplicity"]==1]  #?? this is the cut for singles
        

        # Setup and load grid with data. If data is not cached, then cache data from whole grid into voxel dataframes.
        voxel_grid = RPMeoGrid(radius_grid_array, period_grid_array, mass_grid_array, eccentricity_grid_array, omega_grid_array)
        voxel_grid.setup_dataframes(df.columns)


        print("initialized voxel grid!")

        # Possible -- read in gaia data (though I don't think we need this right now)
        # gaia_df = pd.read_csv(runprops["gaia_data_filename"],delimiter='\t',header=1,engine='pyarrow')
        # gaia_df = gaia_df[["KIC","Mass","Teff","Rad"]]
        

        print("starting stellar df")
        # Read in the stellar df
        stellar_df = pd.read_csv(runprops["stellar_data_filename"],engine='pyarrow',delimiter='\t') # used to be from ../data/keplerstellar.csv, now is from Berger et al 2020

        # Read in the expanded stellar df, which has CDPP values
        rowe_stellar_df = pd.read_csv(runprops["rowe_stellar_data_filename"],engine='pyarrow') # this is the stellar data from Rowe et al 2015.
        rowe_stellar_df = rowe_stellar_df[rowe_stellar_df["st_delivname"]=="q1_q17_dr25_stellar"]

        print("len(stellar_df) before cuts: ",len(stellar_df))

        # Make the cuts to stellar catalog based off of temperature, logg
        stellar_df = stellar_df[(stellar_df["Teff"]>4000) & (stellar_df["Teff"]<7000)]
        stellar_df = stellar_df[(stellar_df["logg"]>4)]

        print("len(stellar_df) after hsu cuts: ",len(stellar_df))

        # Remove the stellar entries with nans in mass or temperature
        stellar_df = stellar_df[(~stellar_df["Mass"].isna())  & (~stellar_df["Teff"].isna())] # & (~stellar_df["limbdark_coeff1"].isna())]

        # Merge with the Rowe catalog to get the CDPP and dataspan values for each star
        stellar_df = stellar_df.merge(
                                    rowe_stellar_df[
                                        ['kepid', 'dataspan', 'rrmscdpp01p5', 'rrmscdpp02p0', 'rrmscdpp02p5', 'rrmscdpp03p0',
                                        'rrmscdpp03p5', 'rrmscdpp04p5', 'rrmscdpp05p0', 'rrmscdpp06p0', 'rrmscdpp07p5',
                                        'rrmscdpp09p0', 'rrmscdpp10p5', 'rrmscdpp12p0', 'rrmscdpp12p5', 'rrmscdpp15p0']
                                    ],
                                    left_on='KIC',
                                    right_on='kepid',
                                    how='left'   # keeps all rows from stellar_df
                                )
        
        print("len(stellar_df) after removing nans: ",len(stellar_df))
        
        print("stellar df cuts applied")

        # Get the KICs of the stars in the stellar df
        stellar_df_kics = stellar_df["KIC"]
        # Only include the planets in the multis df that are in the stellar df
        df = df[df["KIC"].isin(stellar_df_kics)]
        # Only include the planets in the singles that are in the stellar df
        singles_dr_df = singles_dr_df[singles_dr_df["kepid"].isin(stellar_df_kics)]

        print("len(singles_dr_df): ",len(singles_dr_df))
        print("nan in singles_dr_df periods: ",singles_dr_df["koi_period"].isna().sum())
        print("nan in singles_dr_df periods err1: ",singles_dr_df["koi_period_err1"].isna().sum())
        print("nan in singles_dr_df periods err2: ",singles_dr_df["koi_period_err2"].isna().sum())

        print("location of nan in errors: ",singles_dr_df[singles_dr_df["koi_period_err1"].isna() | singles_dr_df["koi_period_err2"].isna()]["kepid"])

        # Remove the planets in the singles df that have nans in their period errors, since we need these for sampling the posteriors
        singles_dr_df = singles_dr_df[~(singles_dr_df["koi_period_err1"].isna() | singles_dr_df["koi_period_err2"].isna())]
        # Reset the index so we can iterate through singles df
        singles_dr_df = singles_dr_df.reset_index(drop=True)
        # Give the singles df the same cols as the multis df, sample ecc and omega for the singles
        processed_singles_dr_df = process_singles_df(singles_dr_df,stellar_df)
        
        # Remove the planets with densities above or below a certain threshold, because they are unphysical
        if runprops["exclude_bad_densities"]:
            df = df[(df["rho_p"]<runprops["maximum_density"]) & (df["rho_p"]>runprops["minimum_density"])]


        print("length of df after before requiring stability: ",len(df))
        # Exclude any posterior draw that has a periastron less than 2 stellar radii
        sm_axis = (df["Period_days"] * 24 * 60 * 60)**(2/3) * (G / (4*np.pi**2))**(1/3) * (df["M_pE"] * MEKG + stellar_df.set_index("KIC").loc[df["KIC"],"Mass"].values * MSKG)**(1/3)
        periapsis = (1 - df["e"]) * sm_axis
        df = df[periapsis >= 2 * stellar_df.set_index("KIC").loc[df["KIC"],"Rad"].values]
        print("length of df after after requiring stability: ",len(df))

        print("length of singles df after before requiring stability: ",len(processed_singles_dr_df))
        sm_axis = (processed_singles_dr_df["Period_days"] * 24 * 60 * 60)**(2/3) * (G / (4*np.pi**2))**(1/3) * (processed_singles_dr_df["M_pE"] * MEKG + stellar_df.set_index("KIC").loc[processed_singles_dr_df["kepid"],"Mass"].values * MSKG)**(1/3)
        periapsis = (1 - processed_singles_dr_df["e"]) * sm_axis
        processed_singles_dr_df = processed_singles_dr_df[periapsis >= 2 * stellar_df.set_index("KIC").loc[processed_singles_dr_df["kepid"],"Rad"].values]
        print("length of processed_singles_dr_df after requiring stability: ",len(processed_singles_dr_df))


        # Define the "unique planet" column which is the combo of the KIC and planet number
        df['unique_planet'] = df['KIC'].astype(str) + "_" + df['planet'].astype(str)
        # Make a dict of how many unique planets there are, with the unique planet as the key and the count as the value
        kic_dict_multis = df['unique_planet'].value_counts().to_dict()

        print("kic_dict_multis: ", kic_dict_multis)

        # List of unique planets that have less than 50 samples, which we remove because they will be improperly weighted
        unique_planet_to_remove = [k for k, v in kic_dict_multis.items() if v < 50]

        # Actually remove these planets
        for unique_planet in unique_planet_to_remove:
            df = df[df['unique_planet'] != unique_planet]

        # Update the kic_dict_multis after removing the planets with less than 50 samples (necessary for reweighting procedure)
        kic_dict_multis = {k: v for k, v in kic_dict_multis.items() if v >= 50}

        print("kic_dict_multis: ", kic_dict_multis)
        
        # Give the singles a unique planet column as well
        processed_singles_dr_df['unique_planet'] = processed_singles_dr_df['kepid'].astype(int).astype(str) + "_" + "0.1"

        # Make a dict of how many unique planets are in singles
        kic_dict_singles = processed_singles_dr_df['unique_planet'].value_counts().to_dict()

        # List of unique planets that have less than 50 samples, which we remove because they will be improperly weighted
        unique_planet_to_remove = [k for k, v in kic_dict_singles.items() if v < 50]

        # Actually remove these planets
        for unique_planet in unique_planet_to_remove:
            processed_singles_dr_df = processed_singles_dr_df[processed_singles_dr_df['unique_planet'] != unique_planet]

        # Update the kic_dict_singles after removing the planets with less than 50 samples (necessary for reweighting procedure)
        kic_dict_singles= {k: v for k, v in kic_dict_singles.items() if v >= 50}

        print("kic_dict_singles: ", kic_dict_singles)

        # Merge the multis and singles count dicts
        kic_dict = kic_dict_multis | kic_dict_singles

        print("kic_dict: ", kic_dict)

        # Give the RPMeoGrid the kic_dict so that it can reweight the planets
        voxel_grid.set_kic_dict(kic_dict)

        # Concatenate the multis and processed singles dfs together to get the KDC - Kepler Dynamical Catalog
        final_kdc_df = pd.concat([df, processed_singles_dr_df.rename(columns={"kepid":"KIC"})], ignore_index=True)
        ######## ADD A FLAG TO SEE IF ITS A SINGLE OR A MULTI (FOR PLOTTING PURPOSES)

        print("final_kdc_df: ",final_kdc_df)

        print("final_kdc_df columns: ",final_kdc_df.columns)
        
        print("length of df after matching to stellar catalog, filtering densities: ",len(final_kdc_df))

        # Add the data to the RPMeoGrid voxel grid object (this object will be written to a json, then read in for the model runs)
        voxel_grid.add_data(final_kdc_df)

        # Create a small stellar df with 100 random stars, to set up the completeness grid. (could be expanded to entire stellar catalog)
        stellar_df_reduced=stellar_df.sample(n=400,random_state=22)


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

        import pyarrow as pa
        import pyarrow.csv as csv

        stellar_table = pa.Table.from_pandas(stellar_df)
        csv.write_csv(stellar_table, "../data/keplerstellar_with_cuts.csv")

        # stellar_df.to_csv("../data/keplerstellar_with_cuts.csv")

        final_kdc_table = pa.Table.from_pandas(final_kdc_df)
        csv.write_csv(final_kdc_table, "../data/final_kdc.csv")

        # final_kdc_df.to_csv("../data/final_kdc.csv")

        final_kdc_df_columns = json.dumps(list(final_kdc_df.columns))
        with open('../data/dataframe_column_names.json', "w") as f:
            f.write(final_kdc_df_columns)
        
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