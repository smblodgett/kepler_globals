# defines the grid dimensions for the grid analysis, outputs a list of the parameters for the grid, maybe makes an initial guess for each one?

import numpy as np
import pandas as pd

class RPM_Voxel:
    
    def __init__(self,bottom_radius,top_radius,bottom_period,top_period,bottom_mass,top_mass):
        
        self.bottom_radius = bottom_radius
        self.top_radius = top_radius
        self.bottom_period = bottom_period
        self.top_period = top_period
        self.bottom_mass = bottom_mass
        self.top_mass = top_mass
        self.df = pd.DataFrame()
    
    def within(self,radius,period,mass):
        return radius >= bottom_radius and radius < top_radius and period >= bottom_period and period < top_period and mass >= bottom_mass and mass < top_mass 
    
    def add_data(self): # need to add in each invididual data point to different voxels
        pass
    
    def __str__(self):
        return (f"R: {self.bottom_radius} - {self.top_radius}, "
                f"P: {self.bottom_period} - {self.top_period}, "
                f"M: {self.bottom_mass} - {self.top_mass}")
    
    
class RPM_Grid:

    def __init__(self,radius_grid_array,period_grid_array,mass_grid_array):
        
        r_len = len(radius_grid_array) - 1
        p_len = len(period_grid_array) - 1
        m_len = len(mass_grid_array) - 1

        self.voxel_array = [[[RPM_Voxel(radius_grid_array[i],radius_grid_array[i+1],period_grid_array[j],period_grid_array[j+1],mass_grid_array[k],mass_grid_array[k+1]) for k in range(m_len)] for j in range(p_len)] for i in range(r_len)]

        

# for empty voxels, we need to put a slightly nonuniform prior on them so that it doesn't go crazy...it should just return the priors for these ones.
            
