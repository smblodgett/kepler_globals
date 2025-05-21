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
import h5py
import emcee
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
        return len(self.df) if hasattr(self,"df") else 0
    
    def cache_data(self,cache_path):
        self.df.to_csv(cache_path+f"/voxel_{self.id_number}.csv") ## could indicate with... _R{self.bottom_radius}-{self.top_radius}_P{self.bottom_period}-{self.top_period}_M{self.bottom_mass}-{self.top_mass} ??
        
    def get_cached_data(self,cache_path):
        self.df = pd.read_csv(cache_path+f"/voxel_{self.id_number}.csv",index_col=0)

    def get_cached_data_count(self,cache_path):
        with open(cache_path+f"/voxel_{self.id_number}.csv", 'r', encoding='utf-8') as f:
          return sum(1 for _ in f) - 1  # subtract header
        
    def is_implausible(self):
        def density(radius, mass):
            assert radius >= 0
            return (mass * MEG) / ((4/3)*np.pi*(RECM*radius)**3) if radius > 0 else np.inf
        return density(self.top_radius,self.bottom_mass) > 30 or density(self.bottom_radius,self.top_mass) < 0.01 

    def get_Rmrp(self,backend_path="../results/backend"):
        # print(self)
        h5_file = f"/voxel_{self.id_number}_chain.h5"
        if not os.path.exists(backend_path+h5_file):
            return 0.0,0.0,0.0,self.bottom_radius, self.top_radius, self.bottom_period, self.top_period, self.bottom_mass, self.top_mass
        
        size_bytes = os.path.getsize(backend_path+h5_file)
        size_mb = size_bytes / (1024 ** 2)
        if size_mb < 1: ### figure out a better way to check if it's empty???
            return 0.0,0.0,0.0,self.bottom_radius, self.top_radius, self.bottom_period, self.top_period, self.bottom_mass, self.top_mass
        
        reader = emcee.backends.HDFBackend(backend_path + h5_file)
        samples = np.array(reader.get_chain())
        # print("samples.shape, samples",samples.shape,samples)
        better_samples = samples.reshape(-1,1)
        # print("better samples",better_samples)
        # input()
        mean = np.mean(better_samples)
        lower = np.percentile(better_samples, 15.87)
        upper = np.percentile(better_samples, 84.13)
        # print("mean, lower, and upper: ",mean, lower, upper)
        return mean, lower, upper, self.bottom_radius, self.top_radius, self.bottom_period, self.top_period, self.bottom_mass, self.top_mass
    
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
        self.voxel_array = np.array(self.voxel_array,dtype=object)
        id_number=0
        self.id_array=np.empty((self.r_len,self.p_len,self.m_len))

        it = np.nditer(self.id_array, flags=['multi_index'], op_flags=['writeonly'])
        for id_number in range(self.r_len * self.p_len * self.m_len):
            i, j, k = it.multi_index  # Gives current (i, j, k) position
            self.voxel_array[i, j, k].create_id(id_number)
            it[0] = id_number  # Write to id_array
            it.iternext()

        # i_idx = 0
        # for i in self.voxel_array:
        #     j_idx = 0
        #     for j in i:
        #         k_idx=0
        #         for k in j:
        #             k.create_id(id_number)
        #             self.id_array[i_idx,j_idx,k_idx] = id_number  #### this needs to be fixed: look at type of i,j for quick find by id
        #             id_number+=1
        #             k_idx+=1
        #         j_idx+=1
        #     i_idx+=1
        

    def setup_dataframes(self,columns,voxel_id,is_cached):
        if is_cached:
            self.find_voxel_by_id(voxel_id).setup_dataframe(columns)
        else:
            for voxel in self.voxel_array.flat:
                voxel.setup_dataframe(columns)
                        
    def add_data(self,df):
        
        r_idx = np.searchsorted(self.radius_grid_array, df['R_pE'].values, side='right') - 1
        p_idx = np.searchsorted(self.period_grid_array, df['Period_days'].values, side='right') - 1
        m_idx = np.searchsorted(self.mass_grid_array, df['M_pE'].values, side='right') - 1

        # Filter valid entries
        valid_mask = (
            (r_idx >= 0) & (r_idx < self.r_len) &
            (p_idx >= 0) & (p_idx < self.p_len) &
            (m_idx >= 0) & (m_idx < self.m_len)
        )
        df_valid = df.loc[valid_mask].copy()
        df_valid['r_idx'] = r_idx[valid_mask]
        df_valid['p_idx'] = p_idx[valid_mask]
        df_valid['m_idx'] = m_idx[valid_mask]

        # Sort to improve memory access pattern (optional, measurable on big data)
        df_valid.sort_values(['r_idx', 'p_idx', 'm_idx'], inplace=True)

        # Grouping via numpy keys instead of tuple hashing (faster)
        index_array = df_valid[['r_idx', 'p_idx', 'm_idx']].values
        voxel_keys, inverse = np.unique(index_array, axis=0, return_inverse=True)

        # Slice records into voxel groups without Python dicts
        for voxel_id, (r, p, m) in enumerate(voxel_keys):
            group_mask = (inverse == voxel_id)
            df_chunk = df_valid.loc[group_mask].drop(columns=['r_idx', 'p_idx', 'm_idx'])
            self.voxel_array[r, p, m].add_data(df_chunk)


        # df['r_idx'] = pd.cut(df['R_pE'], bins=self.radius_grid_array, labels=False, include_lowest=True, right=False)
        # df['p_idx'] = pd.cut(df['Period_days'], bins=self.period_grid_array, labels=False, include_lowest=True, right=False)
        # df['m_idx'] = pd.cut(df['M_pE'], bins=self.mass_grid_array, labels=False, include_lowest=True, right=False)

        # # Drop rows outside of grid (NaNs)
        # df_clean = df.dropna(subset=['r_idx', 'p_idx', 'm_idx']).copy()
        # df_clean[['r_idx', 'p_idx', 'm_idx']] = df_clean[['r_idx', 'p_idx', 'm_idx']].astype(int)

        # # Group rows by voxel indices
        # for (r_idx, p_idx, m_idx), group in df_clean.groupby(['r_idx', 'p_idx', 'm_idx']):
        #     voxel = self.voxel_array[r_idx,p_idx,m_idx]
        #     voxel.add_data(group.drop(['r_idx', 'p_idx', 'm_idx'], axis=1))
            
            
    def cache_dataframes(self,cache_path="../data/thinned/voxel_data"):
        for voxel in self.voxel_array.flat:
            voxel.cache_data(cache_path)
    
    
    def find_voxel_by_id(self,voxel_id):
        location = np.argwhere(self.id_array == voxel_id)
        if location.size == 0:
            print("Unable to find voxel with given voxel id.")
            return None
        i, j, k = location[0]
        return self.voxel_array[i][j][k]
        

    def find_voxel_by_coordinates(self,radius,period,mass):
        for voxel in self.voxel_array.flat:
            if voxel.within(radius,period,mass):
                return voxel
                    
        print("unable to find voxel with given coordinates.")
        
    def count_points_in_RP_column(self,high_radius,low_radius,high_period,low_period,cache_path,is_cached=True):
        num_points = 0
        for voxel in self.voxel_array.flat:
            if voxel.bottom_radius == low_radius and voxel.top_radius == high_radius and voxel.bottom_period == low_period and voxel.top_period == high_period:
                if is_cached: 
                    length_voxel = voxel.get_cached_data_count(cache_path)
                else:
                    length_voxel = len(voxel.df)
                num_points += length_voxel
                        
        return num_points
    
    def make_mass_divided_weights(self,voxel_id,cache_path,is_cached=True):
        for voxel in self.voxel_array.flat:
            if voxel.id_number == voxel_id:
              voxel_number_of_posterior_draws = voxel.num_data()
              voxel.df["mass_divided_weights"] = voxel.df['occurrence_rate_hsu']  * voxel_number_of_posterior_draws / self.count_points_in_RP_column(voxel.top_radius,voxel.bottom_radius,voxel.top_period,voxel.bottom_period,cache_path,is_cached) # used to multiply by voxel_number
        
    def get_occurrence_rate(self,voxel_id):
        voxel = self.find_voxel_by_id(voxel_id)
        if len(voxel.df) != 0:
            return voxel.df["occurrence_rate_hsu"].iloc[0]
        else:
            occurrence_rate_df = pd.read_csv("../data/occurrence_rates_hsu.csv")
            mask = ((occurrence_rate_df["radius_lower"] == voxel.bottom_radius) &
                    (occurrence_rate_df["radius_upper"] == voxel.top_radius) &
                    (occurrence_rate_df["period_lower"] == voxel.bottom_period) &
                    (occurrence_rate_df["period_upper"] == voxel.top_period)
                    )
            return occurrence_rate_df.loc[mask].iloc[0]["occurrence"]
        
    def get_Rmrps(self):
        
        self.Rmrp_array = np.empty((self.r_len, self.p_len, self.m_len, 9))

        for i in range(self.r_len):
            for j in range(self.p_len):
                for k in range(self.m_len):
                    self.Rmrp_array[i, j, k] = self.voxel_array[i, j, k].get_Rmrp()
        
        # print(self.Rmrp_array.reshape(-1,9))
        return self.Rmrp_array.reshape(-1,9)       
                

        
        #     for lookup_voxel in self.voxel_array.flat:
        #         if lookup_voxel.bottom_radius == voxel.bottom_radius and lookup_voxel.top_radius == voxel.top_radius and lookup_voxel.bottom_period == voxel.bottom_period and lookup_voxel.top_period == voxel.top_period:
        #             print("bottom radius: ",voxel.bottom_radius)
        #             print("top radius: ",voxel.top_radius)
        #             print("bottom period: ",voxel.bottom_period)
        #             print("top period: ",voxel.top_period)

        #             luv_id = lookup_voxel.id_number
        #             print("HHHEEERREEEE" + f" with {luv_id}")

        #             lines = 0
        #             with open(f"../data/thinned/voxel_data/voxel_{luv_id}.csv", 'rb') as f:
        #                 print("hshsfhsdfhsadfhasdhsfhsdfhhdssdhsdfh")
        #                 lines = sum(1 for line in f) - 1

        #             if lines > 0:
        #                 print(f"voxel {luv_id} had lines!!!!!")
        #                 return lookup_voxel.df["occurrence_rate_hsu"].iloc[0]
                    
        # print("Unable to get occurrence rate.")


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