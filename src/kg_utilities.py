import numpy as np
import commentjson as json
from scipy.integrate import quad, simpson
from scipy.special import gamma
from kg_constants import *

def radius_given_density_mass(density,mass):
    return (((mass)*MEG)/((4/3)*np.pi*density))**(1/3) / RECM


def mass_given_density_radius(density,radius):
    return ((4/3)*np.pi*density/MEG)*(radius * RECM)**3


# def get_detection_probability(MES,a=29.14,b=0.284,c=0.891):
#     def integrand(x):
#         return (c / (b**a * gamma(a)) ) * x**(a-1) * np.exp(-x/b)
#     return quad(integrand, 0, MES)


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
        print('reading in the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        """Return the parsed Json dictionary."""
        return self.data