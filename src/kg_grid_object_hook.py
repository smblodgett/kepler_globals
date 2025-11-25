import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator
from kg_griddefiner import RPMeoVoxel, RPMeoGrid



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
        "voxel_array","completeness_array","id_array"
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
        grid.completeness_array  = np.array(dct["completeness_array"])
        grid.id_array            = np.array(dct["id_array"])
        grid.likelihood_array    = np.array(dct["likelihood_array"])

        # 2c) Rebuild the interpolators so .p_detection_interp exists
        grid.completeness_interp = RegularGridInterpolator(
            (grid.radius_grid_array,
            grid.period_grid_array,
            grid.mass_grid_array,
            grid.eccentricity_grid_array,
            grid.omega_grid_array),
            grid.completeness_array
        )

        return grid

    # fallback: leave dict alone
    return dct