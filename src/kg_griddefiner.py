"""
kg_griddefiner.py
=================

Contains several classes useful for creating exoplanet population models.

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

from kg_constants import *
from kg_utilities import radius_given_density_mass, mass_given_density_radius, num_data_with_weighting
from kg_probability_distributions import get_MES, get_transit_probability, get_detection_probability_hsu
from scipy.interpolate import RegularGridInterpolator


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
        self.id_number = -1 # Set to -1 as a flag that the voxel has been initialized but not assigned an id value within a full grid.
        self.num_excluded_by_priors = 0 # Number of posterior draws in voxel that were excluded by the priors.
        self.is_add_data = False # Flag to indicate whether the voxel has data added to it.
    
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
        frames = [df for df in [self.df, df_chunk] if not df.empty and not df.isna().all().all()]
    
        if frames:
            self.df = pd.concat(frames, ignore_index=True)
        else:
            self.df = pd.DataFrame()  # Ensure self.df is still a valid DataFrame

        self.is_add_data = True

    def update_excluded_by_priors(self,upper_density_limit=30,lower_density_limit=0.01):    
        assert self.is_add_data, "Data must be added to the voxel before updating excluded draws by priors."
        self.num_excluded_by_priors = len(self.df) - self.num_data(upper_density_limit,lower_density_limit)  # Update the number of excluded draws by priors
        
    def num_data(self,upper_density_limit=30,lower_density_limit=0.01):
        """
        Returns the number of rows/posterior draws within the voxel.
        If a posterior is outside of the specified density limits, it is excluded from the count.
        """
        
        mask = ((self.df["R_pE"] <= radius_given_density_mass(lower_density_limit, self.df['M_pE'])) & 
                (self.df["R_pE"] >= radius_given_density_mass(upper_density_limit, self.df['M_pE'])) & 
                (self.df["M_pE"] <= mass_given_density_radius(upper_density_limit, self.df['R_pE'])) &
                (self.df["M_pE"] >= mass_given_density_radius(lower_density_limit, self.df['R_pE']))
                )
        if hasattr(self, "df") and (df_length := len(self.df[mask])) != 0:
            return df_length
        return 0
    
    
    def cache_data(self,cache_path):
        """Saves the dataframe of a voxel to a csv identified with the voxel's id number."""
        self.df.to_csv(cache_path+f"/voxel_{self.id_number}.csv")
        
    def get_cached_data(self,cache_path):
        """Loads the data that was cached into the voxel's dataframe."""
        self.df = pd.read_csv(cache_path+f"/voxel_{self.id_number}.csv",index_col=0)

    def get_cached_data_count(self,cache_path):
        """Reads the line count from saved csv for a voxel (minus 1), indicating the number of posterior draws for that voxel"""
        with open(cache_path+f"/voxel_{self.id_number}.csv", 'r', encoding='utf-8') as f:
          return sum(1 for _ in f) - 1  # subtract header
        
    def get_cached_num_excluded_by_priors(self,cache_path,upper_density_limit=30,lower_density_limit=0.01):
        """Reads the number of posterior draws that were excluded by the priors from a cached csv."""
        excluded_df = pd.read_csv(cache_path + f"/excluded_by_priors_{upper_density_limit}_{lower_density_limit}.csv")
        if self.id_number in excluded_df["voxel_id"].values:
            return excluded_df.loc[excluded_df["voxel_id"] == self.id_number, "excluded_count"].values[0]
        else:
            return 0    
        
    def is_implausible(self,upper_density_limit=30,lower_density_limit=0.01):
        """Returns true if a voxel is implausible, i.e., outside of the 0.01-30 g/cm3 density priors which mark the physical limits of planetary composition."""
        def density(radius, mass): # Returns density in g/cm3 given a radius in Earth radii and mass in Earth masses.
            assert radius >= 0
            return (mass * MEG) / ((4/3)*np.pi*(RECM*radius)**3) if radius > 0 else np.inf
        return density(self.top_radius,self.bottom_mass) > upper_density_limit or density(self.bottom_radius,self.top_mass) < lower_density_limit

    def get_Rmrp(self,nburnin,backend_path="../results/backend"):
        """
        Calculates several statistical metrics from the nonparametric grid model run on a voxel's occurrence rate.

        Parameters
        ----------
        backend_path : str, optional
          The pathway to the emcee backend (results from running the emcee model). Default='../results/backend'
        
        Returns
        -------
        mean : float
          The mean value of Rmrp from the emcee model. If no backend file is found, 
          or the file was created but unfilled, the returned mean = 0.0
        lower : float
          The -2σ location of Rmrp. Equals 0.0 if no backend file is found, or 
          the file contains no data.
        upper : float
          The 2σ locatoin of Rmrp.Equals 0.0 if no backend file is found, or 
          the file contains no data.
        The voxel's boundaries and id are then returned for scripting convenience in kg_plots.
        """
        h5_file = f"/voxel_{self.id_number}_chain.h5"
        if not os.path.exists(backend_path+h5_file):
            return 0.0,0.0,0.0,self.bottom_radius, self.top_radius, self.bottom_period, self.top_period, self.bottom_mass, self.top_mass, self.id_number
        
        size_bytes = os.path.getsize(backend_path+h5_file)
        size_mb = size_bytes / (1024 ** 2)
        if size_mb < 1: ### figure out a better way to check if it's empty???
            return 0.0,0.0,0.0,self.bottom_radius, self.top_radius, self.bottom_period, self.top_period, self.bottom_mass, self.top_mass, self.id_number
        
        reader = emcee.backends.HDFBackend(backend_path + h5_file)
        samples = np.array(reader.get_chain())
        samples = samples[nburnin:,:,:]
        better_samples = samples.reshape(-1,1)
        mean = np.mean(better_samples)
        lower = np.percentile(better_samples, 15.87)
        upper = np.percentile(better_samples, 84.13)
        return mean, lower, upper, self.bottom_radius, self.top_radius, self.bottom_period, self.top_period, self.bottom_mass, self.top_mass, self.id_number
    
    def __str__(self):
        """Returns a string representation of the voxel."""
        return (f"RPM_Voxel(id: {self.id_number}, R: {self.bottom_radius} - {self.top_radius}, "
                f"P: {self.bottom_period} - {self.top_period}, "
                f"M: {self.bottom_mass} - {self.top_mass})"
                f"number of data points: {self.num_data()}")
    
class RPMeoVoxel(RPMVoxel):
    """
    Represents part of the radius-period-mass-eccentricity exoplanet occurrence space.
    
    This class inherits from RPMVoxel and adds an eccentricity dimension.
    
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
    bottom_eccentricity : float
      The lower limit eccentricity of the voxel.
    top_eccentricity : float
      The upper limit eccentricity of the voxel.
    
    Attributes
    ----------
    Inherits all attributes from RPMVoxel.
    
    """
    def __init__(self,bottom_radius,top_radius,bottom_period,top_period,bottom_mass,top_mass,bottom_eccentricity,top_eccentricity,bottom_omega,top_omega):
        """
        Initializes the RPMeVoxel with the given radius, period, mass, and eccentricity limits.
        """
        super().__init__(bottom_radius,top_radius,bottom_period,top_period,bottom_mass,top_mass)
        self.bottom_eccentricity = bottom_eccentricity
        self.top_eccentricity = top_eccentricity
        self.bottom_omega = bottom_omega
        self.top_omega = top_omega
    
    def within(self,radius,period,mass,eccentricity,omega):
        """For a given value in radius-period-mass-eccentricity space, returns True if the voxel contains this value."""
        return (super().within(radius,period,mass) 
                and eccentricity >= self.bottom_eccentricity and eccentricity < self.top_eccentricity
                and omega >= self.bottom_omega and omega < self.top_omega)
    
    def get_lower_bounds(self):
        return self.bottom_period, self.bottom_mass, self.bottom_radius, self.bottom_eccentricity, self.bottom_omega
    
    def get_upper_bounds(self):
        return self.top_period, self.top_mass, self.top_radius, self.top_eccentricity, self.top_omega
    
    def get_centroid_coordinate(self):
        coords = tuple((lower+upper)/2 for lower,upper in zip(self.get_lower_bounds(), self.get_upper_bounds()))
        return coords
    
    def add_column_name(self, columns):
        if hasattr(self, "df") and len(self.df) > 0:
            self.df.columns = columns
        else:
            self.df = pd.DataFrame(columns=columns)
    
    def __str__(self):
        """Returns a string representation of the voxel."""
        return (f"RPMeo_Voxel(id: {self.id_number}, R: {self.bottom_radius} - {self.top_radius}, "
                f"P: {self.bottom_period} - {self.top_period}, "
                f"M: {self.bottom_mass} - {self.top_mass}, "
                f"e: {self.bottom_eccentricity} - {self.top_eccentricity}, "
                f"omega: {self.bottom_omega} - {self.top_omega}) "
                f"number of data points: {self.num_data()}") 

    
class RPMGrid:
    """
    Represents the full 3D grid of many voxels of radius-period-mass exoplanet occurrence space.
    
    Parameters
    ----------
    radius_grid_array : list(float)
      A list of the radii values denoting the voxel boundaries of the grid.
    period_grid_array : list(float)
      A list of the period values denoting the voxel boundaries of the grid.
    mass_grid_array : list(float)
      A list of the mass values denoting the voxel boundaries of the grid.
    
    Attributes
    ----------
    radius_grid_array : list(float)
      A list of the radii values denoting the voxel boundaries of the grid.
    period_grid_array : list(float)
      A list of the period values denoting the voxel boundaries of the grid.
    mass_grid_array : list(float)
      A list of the mass values denoting the voxel boundaries of the grid.
    r_len : int
      The number of radius bins.
    p_len : int
      The number of period bins.
    m_len : int
      The number of mass bins.
    voxel_array : ndarray(RPMVoxel)
      A 3D NumPy array of shape (r_len,p_len,m_len) containing RPMVoxels.
    id_array : ndarray
      A 3D NumPy array of shape (r_len,p_len,m_len) that tracks the ids of the voxels in voxel_array.
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

    def setup_dataframes(self,columns,voxel_id=0,is_cached=False):
        if is_cached:
            self.find_voxel_by_id(voxel_id).setup_dataframe(columns)
        else:
            for voxel in self.voxel_array.flat:
                voxel.setup_dataframe(columns)
    
    def assign_column_names(self,columns):
        for voxel in self.voxel_array.flat:
            voxel.add_column_name(columns)
                        
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
            # num_data_with_weighting(self.voxel_array[r, p, m].df)  # Update the number of data points with weighting
            # self.voxel_array[r, p, m].create_probability_weighted()

    def cache_dataframes(self,cache_path="../data/thinned/voxel_data"):
        """Saves the dataframes of each voxel of the grid."""
        for voxel in self.voxel_array.flat:
            voxel.cache_data(cache_path)
    
    def cache_prior_excluded_values(self,cache_path="../data/thinned/voxel_data",upper_density_limit=30,lower_density_limit=0.01):
        """Stores the number of posterior draws in each voxel that were excluded by the priors in a cache."""
        excluded_df = pd.DataFrame(columns=["voxel_id","excluded_count"])
        excluded_path = cache_path + f"/excluded_by_priors_{upper_density_limit}_{lower_density_limit}.csv"
        if os.path.exists(excluded_path):
            os.remove(excluded_path)
        
        rows = []
        for voxel in self.voxel_array.flat:
            if voxel.is_add_data:
                voxel.update_excluded_by_priors(upper_density_limit,lower_density_limit)
                excluded_count = voxel.num_excluded_by_priors
            else:
                excluded_count = voxel.num_excluded_by_priors
            # write_header = not os.path.exists(excluded_path)

            rows.append({"voxel_id": voxel.id_number, "excluded_count": excluded_count})
        excluded_df = pd.DataFrame(rows)

            # excluded_df = excluded_df.append({"voxel_id": voxel.id_number, "excluded_count": excluded_count}, ignore_index=True)
        excluded_df.to_csv(excluded_path, index=False,mode='a',header=True)
    
    def find_voxel_by_id(self,voxel_id):
        """Finds the voxel represented by a given id number."""
        location = np.argwhere(self.id_array == voxel_id)
        if location.size == 0:
            print("Unable to find voxel with given voxel id.")
            return None
        i, j, k = location[0]
        return self.voxel_array[i][j][k]

    def find_voxel_by_coordinates(self,radius,period,mass):
        """Finds the voxel that contains a given radius, period, mass value."""
        for voxel in self.voxel_array.flat:
            if voxel.within(radius,period,mass):
                return voxel
                    
        print("unable to find voxel with given coordinates.")
        
    def count_points_in_RP_column(self,high_radius,low_radius,high_period,low_period,cache_path,is_cached=True,upper_density_limit=30,lower_density_limit=0.01):
        """Counts the number of points in every voxel that has the same radius and period values."""
        num_points = 0
        for voxel in self.voxel_array.flat:
            if voxel.is_implausible(upper_density_limit,lower_density_limit):
                continue
            if voxel.bottom_radius == low_radius and voxel.top_radius == high_radius and voxel.bottom_period == low_period and voxel.top_period == high_period:
                if is_cached: 
                    length_voxel = voxel.get_cached_data_count(cache_path) - voxel.get_cached_num_excluded_by_priors(cache_path,upper_density_limit,lower_density_limit)
                else:
                    length_voxel = voxel.num_data(upper_density_limit,lower_density_limit)
                num_points += length_voxel
                        
        return num_points
    
    def make_mass_divided_weights(self,voxel_id,cache_path,is_cached=True,upper_density_limit=30,lower_density_limit=0.01):
    
        for voxel in self.voxel_array.flat:
            if voxel.id_number == voxel_id:
              voxel_number_of_posterior_draws = voxel.num_data(upper_density_limit,lower_density_limit)
              voxel.df["mass_divided_weights"] = (voxel.df['occurrence_rate_hsu']  * voxel_number_of_posterior_draws / 
                                                  self.count_points_in_RP_column(voxel.top_radius,voxel.bottom_radius,voxel.top_period,voxel.bottom_period,cache_path,
                                                                                 is_cached,upper_density_limit=upper_density_limit,lower_density_limit=lower_density_limit
                                                                                 )
                                                  ) # used to multiply by voxel_number
        
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
        
    def get_Rmrps(self,nburnin,backend_path="../results/backend"):
        
        self.Rmrp_array = np.empty((self.r_len, self.p_len, self.m_len, 10))

        for i in range(self.r_len):
            for j in range(self.p_len):
                for k in range(self.m_len):
                    self.Rmrp_array[i, j, k] = self.voxel_array[i, j, k].get_Rmrp(nburnin,backend_path=backend_path)
        
        return self.Rmrp_array.reshape(-1,10)       
                
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

class RPMeoGrid(RPMGrid):
    """
    A subclass of RPMGrid that is specifically designed for the radius-period-mass-eccentricity-argument of periapse parametric models.
    
    This class inherits from RPMGrid.
    
    Parameters
    ----------
    radius_grid_array : list(float)
      A list of the radii values denoting the voxel boundaries of the grid.
    period_grid_array : list(float)
      A list of the period values denoting the voxel boundaries of the grid.
    mass_grid_array : list(float)
      A list of the mass values denoting the voxel boundaries of the grid.
    eccentricity_grid_array : list(float)
      A list of the eccentricity values denoting the voxel boundaries of the grid.
    
    Attributes
    ----------
    Inherits all attributes from RPMGrid.
    
    """
    def __init__(self,radius_grid_array,period_grid_array,mass_grid_array,eccentricity_grid_array,omega_grid_array):
        """
        Initializes the RPMeGRID with the given radius, period, mass, and eccentricity grid arrays.
        """
        self.radius_grid_array = radius_grid_array
        self.period_grid_array = period_grid_array
        self.mass_grid_array = mass_grid_array
        self.eccentricity_grid_array = eccentricity_grid_array
        self.omega_grid_array = omega_grid_array
  
        self.r_len = len(radius_grid_array) - 1
        self.p_len = len(period_grid_array) - 1
        self.m_len = len(mass_grid_array) - 1
        self.e_len = len(eccentricity_grid_array) - 1
        self.o_len = len(omega_grid_array) - 1

        self.voxel_array = [[[[[RPMeoVoxel(self.radius_grid_array[i],self.radius_grid_array[i+1],
                                          self.period_grid_array[j],self.period_grid_array[j+1],
                                          self.mass_grid_array[k],self.mass_grid_array[k+1],
                                          self.eccentricity_grid_array[l],self.eccentricity_grid_array[l+1],
                                          self.omega_grid_array[m],self.omega_grid_array[m+1]) 
                                        for m in range(self.o_len)]
                                        for l in range(self.e_len)] for k in range(self.m_len)] 
                                        for j in range(self.p_len)] for i in range(self.r_len)]
        self.voxel_array = np.array(self.voxel_array,dtype=object)
        id_number=0
        self.id_array=np.empty((self.r_len,self.p_len,self.m_len,self.e_len,self.o_len))
        self.p_detection_array = np.empty((self.r_len+1,self.p_len+1,self.m_len+1,self.e_len+1,self.o_len+1))
        self.p_transit_array = np.empty((self.r_len+1,self.p_len+1,self.m_len+1,self.e_len+1,self.o_len+1))

        it = np.nditer(self.id_array, flags=['multi_index'], op_flags=['writeonly'])
        for id_number in range(self.r_len * self.p_len * self.m_len * self.e_len * self.o_len):
            i, j, k, l, m = it.multi_index  # Gives current (i, j, k, l) position
            self.voxel_array[i, j, k, l, m].create_id(id_number)
            it[0] = id_number  # Write to id_array
            it.iternext()

    def add_data(self,df):
        
        r_idx = np.searchsorted(self.radius_grid_array, df['R_pE'].values, side='right') - 1
        p_idx = np.searchsorted(self.period_grid_array, df['Period_days'].values, side='right') - 1
        m_idx = np.searchsorted(self.mass_grid_array, df['M_pE'].values, side='right') - 1
        e_idx = np.searchsorted(self.eccentricity_grid_array, df['e'].values, side='right') - 1
        o_idx = np.searchsorted(self.omega_grid_array, df['omega'].values, side='right') - 1

        # Filter valid entries
        valid_mask = (
            (r_idx >= 0) & (r_idx < self.r_len) &
            (p_idx >= 0) & (p_idx < self.p_len) &
            (m_idx >= 0) & (m_idx < self.m_len) &
            (e_idx >= 0) & (e_idx < self.e_len) &
            (o_idx >= 0) & (o_idx < self.o_len)
            )
        df_valid = df.loc[valid_mask].copy()
        df_valid['r_idx'] = r_idx[valid_mask]
        df_valid['p_idx'] = p_idx[valid_mask]
        df_valid['m_idx'] = m_idx[valid_mask]
        df_valid['e_idx'] = e_idx[valid_mask]
        df_valid['o_idx'] = o_idx[valid_mask]

        # Sort to improve memory access pattern (optional, measurable on big data)
        df_valid.sort_values(['r_idx', 'p_idx', 'm_idx','e_idx','o_idx'], inplace=True)

        # Grouping via numpy keys instead of tuple hashing (faster)
        index_array = df_valid[['r_idx', 'p_idx', 'm_idx','e_idx','o_idx']].values
        voxel_keys, inverse = np.unique(index_array, axis=0, return_inverse=True)

        # Slice records into voxel groups without Python dicts
        for voxel_id, (r, p, m, e, o) in enumerate(voxel_keys):
            group_mask = (inverse == voxel_id)
            df_chunk = df_valid.loc[group_mask].drop(columns=['r_idx', 'p_idx', 'm_idx','e_idx','o_idx'])
            self.voxel_array[r, p, m, e, o].add_data(df_chunk)
            # num_data_with_weighting(self.voxel_array[r, p, m, e].df)
            # self.voxel_array[r, p, m, e].create_probability_weighted()
    
    def setup_completeness_grid(self,stellar_df,N_SAMPLE_STARS=100):
        

        stellar_df=stellar_df.sample(n=N_SAMPLE_STARS,random_state=41)


        it = np.nditer(self.p_detection_array, flags=['multi_index'], op_flags=['writeonly'])
        for grid_edgepoint_number in range((self.r_len+1) * (self.p_len+1) * (self.m_len+1) * (self.e_len+1) * (self.o_len+1)):
            i, j, k, l, m = it.multi_index  # Gives current (i, j, k, l, m) position

            print("mass input: ", self.mass_grid_array[k])
            print("radius input: ", self.radius_grid_array[i])
            print("period input: ", self.period_grid_array[j])
            print("hdfhsdafhsdfhsdsdaasdfadsf")
            print("self.eccentricity_grid_array[l]: ", self.eccentricity_grid_array[l])
            print("eccentricity input: ", self.eccentricity_grid_array[l])
            print("omega input: ", self.omega_grid_array[m])
            MES,n_transits = get_MES(stellar_df, self.mass_grid_array[k],
                                                    self.radius_grid_array[i],
                                                    self.period_grid_array[j],
                                                    self.eccentricity_grid_array[l],
                                                    self.omega_grid_array[m],
                                                    b=0
                                                    )
            assert MES >= 0, "MES should be non-negative, check stellar_df and input parameters."

            self.p_detection_array[i, j, k, l, m] = get_detection_probability_hsu(MES,n_transits)[0]
            
            self.p_transit_array[i, j, k, l, m] = get_transit_probability(stellar_df, self.mass_grid_array[k],
                                                                            self.radius_grid_array[i],
                                                                            self.period_grid_array[j],
                                                                            self.eccentricity_grid_array[l],
                                                                            self.omega_grid_array[m]
                                                                            )
            # it[0] = grid_edgepoint_number  # Write to id_array
            it.iternext()

        self.p_detection_interp = RegularGridInterpolator((self.radius_grid_array,self.period_grid_array,
                                                  self.mass_grid_array,self.eccentricity_grid_array,
                                                  self.omega_grid_array),self.p_detection_array
                                                  )
        
        
        self.p_transit_interp = RegularGridInterpolator((self.radius_grid_array,self.period_grid_array,
                                                  self.mass_grid_array,self.eccentricity_grid_array,
                                                  self.omega_grid_array),self.p_transit_array
                                                  )

        


    def find_voxel_by_id(self,voxel_id):
        """Finds the voxel represented by a given id number."""
        location = np.argwhere(self.id_array == voxel_id)
        if location.size == 0:
            print("Unable to find voxel with given voxel id.")
            return None
        i, j, k, l, m = location[0]
        return self.voxel_array[i][j][k][l][m]

    def find_voxel_by_coordinates(self,radius,period,mass,eccentricity,omega):
        """Finds the voxel that contains a given radius, period, mass value."""
        for voxel in self.voxel_array.flat:
            if voxel.within(radius,period,mass,eccentricity,omega):
                return voxel
                        

    def __str__(self):
        total = self.r_len * self.p_len * self.m_len * self.e_len* self.o_len
        
        # count how many voxels actually have data
        filled = sum(1 for voxel in self.voxel_array.flat
                       if hasattr(voxel, "num_data") and voxel.num_data() > 0)

        return (
            f"RPMeoGrid:\n"
            f"  dimensions: {self.r_len} R × {self.p_len} P × "
            f"{self.m_len} M × {self.e_len} e × {self.o_len} ω\n"
            f"  total voxels: {total}\n"
            f"  filled voxels: {filled}\n"
        )
