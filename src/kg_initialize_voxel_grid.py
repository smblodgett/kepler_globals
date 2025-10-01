import os
import sys
import math
import numpy as np
import pandas as pd
import json
from kg_utilities import ReadJson
from kg_griddefiner import RPMeoGrid, RPMeoVoxel
from kg_param_boundary_arrays import radius_grid_array, period_grid_array, mass_grid_array, eccentricity_grid_array, omega_grid_array


import json
import numpy as np
import pandas as pd



import json
import numpy as np
import pandas as pd
import math
import numbers

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






def main(runprops):
    
    use_cache = os.path.isdir(runprops["voxel_data_folder"]) and not runprops["reload_KMDC"]

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
        df = df[["R_pE","Period_days","M_pE","e","omega"]]#,"p_trans","MES_rowe"]]
        #df = create_probability_weighted(df)
        df.to_csv(runprops["input_data_folder"]+"/KMDC_RPMeo.csv")
        if runprops["verbose"]: print("data has been cached for future runs! (press enter to continue)")
        # input()
    # Otherwise, you can just read in 1 voxel that has its data cached.    
    else:
        df = pd.read_csv(runprops["input_data_folder"]+"/KMDC_RPMeo.csv",index_col=0,engine='pyarrow')
        if runprops["verbose"]: print("read in cached df")

    print("full data df: ",df)
    

    # Setup and load grid with data. If data is not cached, then cache data from whole grid into voxel dataframes.
    voxel_grid = RPMeoGrid(radius_grid_array, period_grid_array, mass_grid_array, eccentricity_grid_array, omega_grid_array)
    voxel_grid.setup_dataframes(df.columns)
    voxel_grid.add_data(df)

    gaia_df = pd.read_csv(runprops["gaia_data_filename"],delimiter='\t',header=1,engine='pyarrow')
    gaia_df = gaia_df[["KIC","Mass","Teff","Rad"]]

    stellar_df = pd.read_csv(runprops["stellar_data_filename"],engine='pyarrow')
    stellar_df = stellar_df[stellar_df["st_delivname"]=="q1_q17_dr25_stellar"]
    stellar_df = stellar_df.rename(columns={"kepid":"KIC"})


    stellar_df = stellar_df.merge(gaia_df, on='KIC', how='left')

    for old_col,new_col in zip(["teff","mass","radius"],["Teff","Mass","Rad"]):
        stellar_df[old_col] = stellar_df[new_col].combine_first(stellar_df[old_col])


    stellar_df = stellar_df[(stellar_df["teff"]>4000) & (stellar_df["teff"]<7000)]
    stellar_df = stellar_df[(stellar_df["logg"]>4)]

    stellar_df = stellar_df[(~stellar_df["mass"].isna()) & (~stellar_df["limbdark_coeff1"].isna()) & (~stellar_df["teff"].isna())]
    voxel_grid.setup_completeness_grid(stellar_df) # this is the kepler stellar catalog, which has the stellar radii and masses
    voxel_grid.setup_likelihood_grid()
    # MES_grid_plot(voxel_grid.p_detection_interp,voxel_grid.p_transit_interp,runprops["completeness_plot_folder"])
    if runprops["verbose"]: print("MES grid has been set up!")


    grid_string = json.dumps(voxel_grid,cls=GridJSONEncoder)

    
    with open(runprops["voxel_json_filename"], "w") as f:
        f.write(grid_string)

    stellar_df.to_csv("../data/keplerstellar_with_cuts.csv")


    df_columns = json.dumps(list(df.columns))
    with open('../data/dataframe_column_names.json', "w") as f:
        f.write(df_columns)



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