"""
kg_griddefiner.py
=================

Contains several classes useful for creating a non-parametric grid model.

Classes
-------
RPMVoxel
  A representation of a certain part of the radius-period-mass space.
RPMGrid
  A collection of many RPMVoxels that span the width of the radius-period-mass space.

Author
------
Steven Blodgett: <blodgett.steven.m@gmail.com>
Created on: 2024-10-10
"""


import os
import numpy as np
import pandas as pd


RECM=6.378*10**8  # earth radius in cm
MEG=5.9721986*10**27 # earth mass in grams


class RPMVoxel:
    """
    Represents part of the radius-period-mass exoplanet occurrence space.
    
    Parameters
    ----------
    bottom_radius : float
      The lower limit radius of the voxel.
    top_radius : float
      The upper limit radius of the voxel.
    bottom_period : float
      The lower limit period of the voxel.
    top_period : float
      The upper limit period of the voxel.
    bottom_mass : float
      The lower limit mass of the voxel.
    top_mass : float
      The upper limit mass of the voxel.

    Attributes
    ----------
    bottom_radius : float
      The lower limit radius of the voxel.
    top_radius : float
      The upper limit radius of the voxel.
    bottom_period : float
      The lower limit period of the voxel.
    top_period : float
      The upper limit period of the voxel.
    bottom_mass : float
      The lower limit mass of the voxel.
    top_mass : float
      The upper limit mass of the voxel.
    id_number : int
      The id number assigned to a voxel (this is used to perform an inference run on 1 voxel).
      As every voxel should be given a unique positive value, a default of -1 serves as a 
      flag that the voxel was not instantiated correctly.
    initial_guess: float
      The initial guess for what value of the occurrence rate per star of that voxel should be.
      Default is the nonsensical -1.0.
    df : DataFrame
      A dataframe to store the data of each posterior draw from the KMDC within the voxel's limits.
    
    """
    def __init__(self,bottom_radius,top_radius,bottom_period,top_period,bottom_mass,top_mass):
        
        self.bottom_radius = bottom_radius
        self.top_radius = top_radius
        self.bottom_period = bottom_period
        self.top_period = top_period
        self.bottom_mass = bottom_mass
        self.top_mass = top_mass
        self.id_number = -1
        self.initial_guess = -1.0
    
    def within(self,radius,period,mass):
        """For a given value in radius-period-mass space, returns True if the voxel contains this value."""
        return radius >= self.bottom_radius and radius < self.top_radius and period >= self.bottom_period and period < self.top_period and mass >= self.bottom_mass and mass < self.top_mass 
    
    def setup_dataframe(self,columns):
        """Instantiates the dataframe associated with the voxel, giving it the columns of the KMDC."""
        self.df = pd.DataFrame(columns=columns)
        
    def create_id(self,id_number):
        """Updates the voxel's id number."""
        self.id_number = id_number
        
    def add_data(self, df_chunk):
        """Adds posterior draw data to the voxel's dataframe."""
        self.df = pd.concat([self.df, df_chunk], ignore_index=True)
        
    def num_data(self):
        """Returns the number of rows/posterior draws within the voxel."""
        return len(self.df)
    
    def cache_data(self,cache_path):
        self.df.to_csv(cache_path+f"/voxel_{self.id_number}.csv") ## could indicate with... _R{self.bottom_radius}-{self.top_radius}_P{self.bottom_period}-{self.top_period}_M{self.bottom_mass}-{self.top_mass} ??
        
    def get_cached_data(self,cache_path):
        self.df = pd.read_csv(cache_path+f"/voxel_{self.id_number}.csv",index_col=0)

    def get_cached_data_count(self,cache_path):
        with open(cache_path+f"/voxel_{self.id_number}.csv", 'r', encoding='utf-8') as f:
          return sum(1 for _ in f) - 1  # subtract header
    
    def create_initial_guess(self):
        """Creates an initial guess for the occurrence rate."""
        self.initial_guess = np.sum(self.df['mass_divided_weights']) 
    
    def __str__(self):
        """Returns a string representation of the voxel."""
        return (f"RPM_Voxel(id: {self.id_number}, R: {self.bottom_radius} - {self.top_radius}, "
                f"P: {self.bottom_period} - {self.top_period}, "
                f"M: {self.bottom_mass} - {self.top_mass})"
                f"number of data points: {self.num_data()}")
    
    
class RPMGrid:
    """
    Represents the full 3D grid of many voxels of radius-period-mass exoplanet occurrence space.
    
    Parameters
    ----------
    
    Attributes
    ----------
    
    """

    def __init__(self,radius_grid_array,period_grid_array,mass_grid_array):
        
        self.radius_grid_array = radius_grid_array
        self.period_grid_array = period_grid_array
        self.mass_grid_array = mass_grid_array
        
        
        self.r_len = len(radius_grid_array) - 1
        self.p_len = len(period_grid_array) - 1
        self.m_len = len(mass_grid_array) - 1

        self.voxel_array = [[[RPMVoxel(self.radius_grid_array[i],self.radius_grid_array[i+1],self.period_grid_array[j],self.period_grid_array[j+1],self.mass_grid_array[k],self.mass_grid_array[k+1]) for k in range(self.m_len)] for j in range(self.p_len)] for i in range(self.r_len)]
        
        id_number=0
        self.id_array=np.array([[[]]])
        for i in self.voxel_array:
            for j in i:
                k_idx=0
                for k in j:
                    k.create_id(id_number)
                    self.id_array[i,j,k_idx] = id_number  #### this needs to be fixed: look at type of i,j for quick find by id
                    id_number+=1
                    k_idx+=1
        

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
            
            
    def cache_dataframes(self,cache_path="../data/thinned/voxel_data"):
        for i in self.voxel_array:
            for j in i:
                for k in j:
                    k.cache_data(cache_path)
    
    
    def find_voxel_by_id(self,voxel_id):
        location = np.argwhere(self.id_array == voxel_id)
        if location.size == 0:
            print("Unable to find voxel with given voxel id.")
            return None

        # location[0] gets the first match
        # tuple(location[0]) makes it usable as an index
        return self.voxel_array[tuple(location[0])]
        # # for i in self.voxel_array:
        # #     for j in i:
        # #         for k in j:
        # #             if k.id_number == voxel_id:
        # #                 return k
                    
        # # print("unable to find voxel with given voxel id.")

        # location = np.argwhere(self.id_array == voxel_id)
        # return self.voxel_array[np.array[tuple(location.T)]]
        
    def find_voxel_by_coordinates(self,radius,period,mass):
        for i in self.voxel_array:
            for j in i:
                for k in j:
                    if k.within(radius,period,mass):
                        return k
                    
        print("unable to find voxel with given coordinates.")
        
    def count_points_in_RP_column(self,high_radius,low_radius,high_period,low_period,cache_path,is_cached=True):
        num_points = 0
        for i in self.voxel_array:
            for j in i:
                for k in j:
                    if k.bottom_radius == low_radius and k.top_radius == high_radius and k.bottom_period == low_period and k.top_period == high_period:
                        if is_cached: 
                            length_voxel = k.get_cached_data_count(cache_path)
                        else:
                            length_voxel = len(k.df)
                        num_points += length_voxel
                        
        return num_points
    
    def make_mass_divided_weights(self,voxel_id,cache_path,is_cached=True):
        for i in self.voxel_array:
            for j in i:
                for k in j:
                    if k.id_number == voxel_id:
                      voxel_number_of_posterior_draws = k.num_data()
                      k.df["mass_divided_weights"] = k.df['occurrence_rate_hsu']  * voxel_number_of_posterior_draws / self.count_points_in_RP_column(k.top_radius,k.bottom_radius,k.top_period,k.bottom_period,cache_path,is_cached) # used to multiply by voxel_number
        
    def __str__(self):
        """Returns a string representation of the entire grid."""
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

##### for empty voxels, we need to put a slightly nonuniform prior on them so that it doesn't go crazy...it should just return the priors for these ones.
            
