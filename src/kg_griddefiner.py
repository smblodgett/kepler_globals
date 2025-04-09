# defines the grid dimensions for the grid analysis, outputs a list of the parameters for the grid, maybe makes an initial guess for each one?

import os
import numpy as np
import pandas as pd


RECM=6.378*10**8  # earth radius in cm
MEG=5.9721986*10**27 # earth mass in grams


class RPM_Voxel:
    
    def __init__(self,bottom_radius,top_radius,bottom_period,top_period,bottom_mass,top_mass):
        
        self.bottom_radius = bottom_radius
        self.top_radius = top_radius
        self.bottom_period = bottom_period
        self.top_period = top_period
        self.bottom_mass = bottom_mass
        self.top_mass = top_mass
        self.id_number = -1
        self.initial_guess = -1
    
    def within(self,radius,period,mass):
        return radius >= self.bottom_radius and radius < self.top_radius and period >= self.bottom_period and period < self.top_period and mass >= self.bottom_mass and mass < self.top_mass 
    
    def setup_dataframe(self,columns):
        self.df = pd.DataFrame(columns=columns)
        
    def create_id(self,id_number):
        self.id_number = id_number
        
    def add_data(self, df_chunk):
        self.df = pd.concat([self.df, df_chunk], ignore_index=True)
        
    def num_data(self):
        return len(self.df)
    
    def create_initial_guess(self):
        self.initial_guess = np.sum(self.df['occurrence_rate_hsu'])
    
    def __str__(self):
        return (f"RPM_Voxel(id: {self.id_number}, R: {self.bottom_radius} - {self.top_radius}, "
                f"P: {self.bottom_period} - {self.top_period}, "
                f"M: {self.bottom_mass} - {self.top_mass})"
                f"number of data points: {self.num_data()}")
    
    
class RPM_Grid:

    def __init__(self,radius_grid_array,period_grid_array,mass_grid_array):
        
        self.radius_grid_array = radius_grid_array
        self.period_grid_array = period_grid_array
        self.mass_grid_array = mass_grid_array
        
        
        self.r_len = len(radius_grid_array) - 1
        self.p_len = len(period_grid_array) - 1
        self.m_len = len(mass_grid_array) - 1

        self.voxel_array = [[[RPM_Voxel(self.radius_grid_array[i],self.radius_grid_array[i+1],self.period_grid_array[j],self.period_grid_array[j+1],self.mass_grid_array[k],self.mass_grid_array[k+1]) for k in range(self.m_len)] for j in range(self.p_len)] for i in range(self.r_len)]
        
        id_number=0
        for i in self.voxel_array:
            for j in i:
                for k in j:
                    k.create_id(id_number)
                    id_number+=1
        

    def setup_dataframes(self,columns):
        for i in self.voxel_array:
            for j in i:
                for k in j:
                    k.setup_dataframe(columns)
                        
    def add_data(self,df):
        df['r_idx'] = pd.cut(df['R_pE'], bins=self.radius_grid_array, labels=False, include_lowest=True, right=False)
        df['p_idx'] = pd.cut(df['Period_days'], bins=self.period_grid_array, labels=False, include_lowest=True, right=False)
        df['m_idx'] = pd.cut(df['M_pE'], bins=self.mass_grid_array, labels=False, include_lowest=True, right=False)

        # Drop rows outside of grid (NaNs)
        df_clean = df.dropna(subset=['r_idx', 'p_idx', 'm_idx']).copy()
        df_clean[['r_idx', 'p_idx', 'm_idx']] = df_clean[['r_idx', 'p_idx', 'm_idx']].astype(int)

        # Group rows by voxel indices
        for (r_idx, p_idx, m_idx), group in df_clean.groupby(['r_idx', 'p_idx', 'm_idx']):
            voxel = self.voxel_array[r_idx][p_idx][m_idx]
            voxel.add_data(group.drop(['r_idx', 'p_idx', 'm_idx'], axis=1))
            
    
    
    def find_voxel(self,voxel_id):
        for i in self.voxel_array:
            for j in i:
                for k in j:
                    print(type(k),k)
                    if k.id_number == voxel_id:
                        return k
                    
        print("unable to find voxel with given voxel id.")
        
    def __str__(self):
        string_representation = ""
        voxel_count = 0
        filled_voxel_count = 0
        radius_count = 0
        for i in self.voxel_array:
            radius_count+=1
            period_count=0
            for j in i:
                period_count+=1
                mass_count=0
                for k in j:
                    string_representation += str(k) + "\n"
                    if k.num_data() > 0:
                        filled_voxel_count+=1
                    voxel_count += 1
                    mass_count+=1
        string_representation += f"total voxels: {voxel_count}\n"
        string_representation += f"filled voxels: {filled_voxel_count}\n"
        string_representation += f"dimensions: {radius_count} Rbins x {period_count} Pbins x {mass_count} Mbins\n"
        return string_representation

# for empty voxels, we need to put a slightly nonuniform prior on them so that it doesn't go crazy...it should just return the priors for these ones.
            
