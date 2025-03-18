# defines the grid dimensions for the grid analysis, outputs a list of the parameters for the grid, maybe makes an initial guess for each one?

import numpy as np

radius_grid_array = [0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,6,8,12,16] # earth radii
period_grid_array = [0.5,1.25,2.5,5,10,20,40,80,160,320] # days
mass_grid_array = [0.125,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,6,8,10,12,16,32,64,128,256,512,1024,2048,4096,8192] # earth mass


class RPM_Voxel:
    
    def __init__(self,bottom_radius,top_radius,bottom_period,top_period,bottom_mass,top_mass):
        
        self.bottom_radius = bottom_radius
        self.top_radius = top_radius
        self.bottom_period = bottom_period
        self.top_period = top_period
        self.bottom_mass = bottom_mass
        self.top_mass = top_mass
    
    def within(radius,period,mass):
        return radius >= bottom_radius and radius < top_radius and period >= bottom_period and period < top_period and mass >= bottom_mass and mass < top_mass 
    
    def add_data(): # need to add in each invididual data point to different voxels
        pass
    
    def __str__(self):
        return (f"R: {self.bottom_radius} - {self.top_radius}, "
                f"P: {self.bottom_period} - {self.top_period}, "
                f"M: {self.bottom_mass} - {self.top_mass}")
    
    


def create_voxel_array(radius_grid_array,period_grid_array,mass_grid_array):
    
    voxel_array = np.empty((len(radius_grid_array), len(period_grid_array), len(mass_grid_array)), dtype=object)

    for i in range(len(radius_grid_array) - 1):
        for j in range(len(period_grid_array) - 1):
            for k in range(len(mass_grid_array) - 1):
                voxel = RPM_Voxel(
                    radius_grid_array[i], radius_grid_array[i+1], 
                    period_grid_array[j], period_grid_array[j+1], 
                    mass_grid_array[k], mass_grid_array[k+1]
                )
                voxel_array[i, j, k] = voxel
                
    print(voxel_array.size)
    assert len(radius_grid_array)*len(period_grid_array)*len(mass_grid_array) == voxel_array.size
    return voxel_array


# for empty voxels, we need to put a slightly nonuniform prior on them so that it doesn't go crazy...it should just return the priors for these ones.
            
