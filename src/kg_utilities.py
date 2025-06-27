import numpy as np
import commentjson as json
from scipy.integrate import quad, simpson
from scipy.special import gamma
from kg_constants import *

def radius_given_density_mass(density,mass):
    return (((mass)*MEG)/((4/3)*np.pi*density))**(1/3) / RECM

def mass_given_density_radius(density,radius):
    return ((4/3)*np.pi*density/MEG)*(radius * RECM)**3

def detection_probability(MES,a=29.14,b=0.284,c=0.891):
    def integrand(x):
        return (c / (b**a * gamma(a)) ) * x**(a-1) * np.exp(-x/b)
    return quad(integrand, 0, MES)

def simpson_detection_probability(MES,a=29.14,b=0.284,c=0.891,N=500):
    x = np.linspace(0, MES, N)
    integrand = (c / (b**a * gamma(a))) * x**(a-1) * np.exp(-x/b)
    return simpson(integrand, x)

class ReadJson:
    """Read and store the contents of a Json file in a dict."""
    def __init__(self, filename):
        """Load the Json file."""
        print('reading in the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        """Return the parsed Json dictionary."""
        return self.data