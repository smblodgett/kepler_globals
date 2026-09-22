import numpy as np
import pandas as pd
import commentjson as json
from scipy.integrate import quad, simpson
from scipy.special import gamma
from kg_constants import *

def radius_given_density_mass(density,mass):
    """Returns the radius of a planet given its density and mass, with units density in g/cm^3, radius in Earth radii, and mass in Earth masses."""
    return (((mass)*MEG)/((4/3)*np.pi*density))**(1/3) / RECM


def mass_given_density_radius(density,radius):
    """Returns the mass of a planet given its density and radius, with units density in g/cm^3, radius in Earth radii, and mass in Earth masses."""
    return ((4/3)*np.pi*density/MEG)*(radius * RECM)**3


def density_given_mass_radius(mass,radius):
    """Returns the density of a planet given its mass and radius, with units density in g/cm^3, radius in Earth radii, and mass in Earth masses."""
    return ((mass)*MEG)/((4/3)*np.pi*(radius * RECM)**3)


def simpson_detection_probability(MES,a=29.14,b=0.284,c=0.891,N=500):
    x = np.linspace(0, MES, N)
    integrand = (c / (b**a * gamma(a))) * x**(a-1) * np.exp(-x/b)
    return simpson(integrand, x)


def create_probability_weighted(df):
    df["p_detection"] = df["MES_rowe"].apply(lambda mes: simpson_detection_probability(mes))
    return df


def num_data_with_weighting(df,upper_density_limit=30,lower_density_limit=0.01): #### though is this just the hsu occurrence rates? can I just use that?
    mask = ((df["R_pE"] <= radius_given_density_mass(lower_density_limit, df['M_pE'])) & 
            (df["R_pE"] >= radius_given_density_mass(upper_density_limit, df['M_pE'])) & 
            (df["M_pE"] <= mass_given_density_radius(upper_density_limit, df['R_pE'])) &
            (df["M_pE"] >= mass_given_density_radius(lower_density_limit, df['R_pE'])) &
            (df["MES_rowe"] > 0) 
            )
    if len(df[mask]) != 0:
        df["num_weighted_data"] = np.sum((1 / df[mask]["p_detection"]) * (1/df[mask]["p_trans"])) 
    elif len(df) != 0:
        df["num_weighted_data"] = 0


class ReadJson:
    """Read and store the contents of a Json file in a dict."""
    def __init__(self, filename):
        """Load the Json file."""
        # print('reading in the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        """Return the parsed Json dictionary."""
        return self.data


def repair_rowe_df_numeric_columns(rowe_df):
    """
    rowe_table_final.csv has two numeric columns where a batch of rows carry
    two whitespace-jammed-together numbers instead of a clean float --
    'rho*_rowe' (55 rows, all a run of consecutive later-appended KOIs from
    roughly KOI 8339 on) and 'E_Rp_rowe' (4 rows). This is a fixed-width-to-
    CSV conversion artifact (rho* sits right after the unusually long,
    high-precision Kmag_rowe field in the original MRT layout), not a real
    measurement -- e.g. "55   1.48712" for one row's rho*_rowe.

    Cross-checked against an independent physical estimate
    (M*_rowe/R*_rowe**3 * solar density): the LAST whitespace-separated token
    matches that estimate to ~1.7% median error across all 55 rho*_rowe rows,
    while the leading token is unrelated (off by a median factor of >400x --
    values like 55 or 916 g/cm^3 aren't plausible densities for the R*/M*
    those same rows report). E_Rp_rowe's 4 affected rows show the same
    pattern (an absurd ~10-11 digit leading token, a small plausible trailing
    one that lines up with that row's own e_Rp_rowe).

    Recovers the real value from the last token instead of discarding it via
    pd.to_numeric(errors='coerce'), which would otherwise silently NaN out
    real data -- 53 of the 55 rho*_rowe rows are real, kept singles (17
    Rowe-endorsed P candidates, 36 S, 2 F).

    Used by both create_ksdc.py/create_nckmdc.py (via
    kg_initialize_voxel_grid.py, right after each of their own rowe_df reads)
    and kg_subsampler.py's rowe_table_attach() (right after its own
    _cached_read_csv("rowe_table_final.csv", ...) call) -- every place this
    pipeline reads rowe_table_final.csv needs the same repair, since the
    corruption lives in the source file itself, not in how any one script
    reads it.
    """
    rowe_df = rowe_df.copy()

    def _last_token(v):
        if isinstance(v, str):
            parts = v.split()
            if len(parts) > 1:
                try:
                    return float(parts[-1])
                except ValueError:
                    return v  # leave anything unparseable for to_numeric to NaN out
        return v

    for col in ["rho*_rowe", "E_Rp_rowe"]:
        rowe_df[col] = rowe_df[col].apply(_last_token)
        rowe_df[col] = pd.to_numeric(rowe_df[col], errors="coerce")

    return rowe_df
