import numpy as np
from kg_constants import *

def radius_given_density_mass(density,mass):
    return (((mass)*MEG)/((4/3)*np.pi*density))**(1/3) / RECM

def mass_given_density_radius(density,radius):
    return ((4/3)*np.pi*density/MEG)*(radius * RECM)**3