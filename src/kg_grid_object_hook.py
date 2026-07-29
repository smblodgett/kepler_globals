import numpy as np

from kg_griddefiner import RPMeoVoxel, RPMeoGrid

def grid_object_hook(dct):

    # Fast discriminator
    if "bottom_radius" in dct:

        if (
            "top_radius" in dct and
            "bottom_period" in dct and
            "top_period" in dct and
            "bottom_mass" in dct and
            "top_mass" in dct and
            "bottom_eccentricity" in dct and
            "top_eccentricity" in dct and
            "bottom_omega" in dct and
            "top_omega" in dct
        ):

            v = RPMeoVoxel(
                dct["bottom_radius"], dct["top_radius"],
                dct["bottom_period"], dct["top_period"],
                dct["bottom_mass"], dct["top_mass"],
                dct["bottom_eccentricity"], dct["top_eccentricity"],
                dct["bottom_omega"], dct["top_omega"],
            )

            v.id_number = dct.get("id_number", -1)

            if "df" in dct:
                v.df_raw = dct["df"]   # lazy
                v.is_add_data = True

            return v

    # Fast discriminator
    if "voxel_array" in dct:

        if (
            "radius_grid_array" in dct and
            "period_grid_array" in dct and
            "mass_grid_array" in dct and
            "eccentricity_grid_array" in dct and
            "omega_grid_array" in dct and
            "completeness_array" in dct and
            "id_array" in dct
        ):

            grid = RPMeoGrid(
                dct["radius_grid_array"],
                dct["period_grid_array"],
                dct["mass_grid_array"],
                dct["eccentricity_grid_array"],
                dct["omega_grid_array"],
            )

            grid.voxel_array = np.array(dct["voxel_array"], dtype=object)

            grid.completeness_array = np.asarray(
                dct["completeness_array"]
            )

            if "transit_prob_array" in dct:
                grid.transit_prob_array = np.asarray(dct["transit_prob_array"])
            else:
                # Backwards compatibility with voxel_grid.json caches built before
                # the p_det/p_tr split (see RPMeoGrid.transit_prob_array). Falling
                # back to all-ones means the point-process data term's
                # interpolate_transit_probability call is a no-op (rather than
                # silently resurrecting the p_det double-counting this split fixes)
                # until kg_initialize_voxel_grid.py is rerun to populate it for real.
                print("WARNING: voxel_grid.json has no transit_prob_array (stale cache). "
                      "Re-run kg_initialize_voxel_grid.py to get a correct p_tr grid for "
                      "the point-process likelihood's data term.")
                grid.transit_prob_array = np.ones_like(grid.completeness_array)

            grid.id_array = np.asarray(dct["id_array"])

            grid.likelihood_array = np.asarray(
                dct["likelihood_array"]
            )

            # lazy interpolator
            grid.completeness_interp = None

            return grid

    return dct