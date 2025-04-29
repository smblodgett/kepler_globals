'''
November 11, 2024
@author: Steven Blodgett 

Takes all the raw converged PhoDym posterior spaces
which have been randomly subsampled by the 
kg_random_row_selector script
and creates a dataframe, then calculates or appends a 
variety of extra parameters, including
orbital elements, occurrence rates, system comparisons, 
and the entire table of Lissauer et al. 

Returns the dataframs as a large combined csv with all planets as thinned/all_thin.csv.
'''

import pandas as pd
import numpy as np
import os
import re
import sys
import math


RAW_PATH = '/hdd2/backup/danielkj/PhoDyMM_results_final/completed_systems/'   # pathway to directory with raw data
SUBSAMPLED_PATH = '/home/byu.local/smb9564/research/hierarchal_modeling/kepler_globals/data/subsampled_rows/' # pathway to directory 
SAVE_PATH = '/home/byu.local/smb9564/research/hierarchal_modeling/kepler_globals/data/thinned/'   # pathway to directory to save data

METOMS=3.0024584*10**-6 # earth mass to solar mass
METOMJ=0.00314636   # earth mass to jupiter mass
MEG=5.9721986*10**27 # earth mass in grams
MEKG=MEG/1000 # earth mass in kilograms
MSKG=1.9891*10**30 # solar mass in kilograms
MSTOMJ=1047.92612 # solar mass in jupiter masses

RECM=6.378*10**8  # earth radius in cm
RETORS=0.009171 # earth radius to solar radius
RJTORE=11.20814 # jupiter radius to earth radius
RJTORS=0.10045 # jupiter radius to solar radius
RSAU=0.00465047 # solar radius in AU
RJAU=4.676*10**-4 # jupiter radius in AU
RSCM=6.9634*10**10 # solar radius in cm

DTOS=24*3600 # days to seconds
MTOAU=6.68458712*10**-12 # meters to AU

G=6.6743*10**-11 # gravitational constant in sci units


# grab the koi number from filename
def find_koi(name_with_numbers):
    numbers = re.findall(r'\d+', name_with_numbers)
    result = ''.join(numbers)
    return result


# take a df and process it into a more useable form, add calculated parameters (esp. rejection constant) and rename bad columns
def process_dataframe(df,koi):
    
    step_number_col = df['Unnamed: 0']

    df = df.drop('Unnamed: 0', axis=1)
    total_columns = len(df.columns)
    
    n_chunks = (total_columns - 7) // 9
        
    processed_chunks = []
    
    # Iterate through each chunk (9 columns) before the last 7 columns
    for i in range(n_chunks):
        chunk_columns = df.iloc[:, i*9:(i+1)*9]
        constant_columns = df.iloc[:, -7:]
        chunk_df = pd.concat([chunk_columns, constant_columns], axis=1)
        chunk_df = column_rename(chunk_df)
        chunk_df['chisq_rank'] = chunk_df['chisq'].rank(method='min', ascending=True) # chisq ranking
        chunk_df['step_number'] = step_number_col # PhoDymm step number
        chunk_df = calculate_params(chunk_df) # add all calculated parameters
        processed_chunks.append(chunk_df)
    final_df = pd.concat(processed_chunks)

    with pd.option_context('mode.chained_assignment', None):
        final_system_df = system_params(final_df) # add system-wide value and comparison columns
        final_system_df = rowe_table_attach(koi,final_system_df) # add table from Lissauer et al.
        final_system_df = final_system_df.drop("Unnamed: 0", axis=1) # get rid of read-in column
        return final_system_df


# rename the columns of the df to be more legible
def column_rename(df):

    num_columns = len(df.columns)
    for n, column in enumerate(df.columns):
        if num_columns-7 <= n:
            if n%9 == 0:
                print(column)
                assert(column=='M$_s$')
                df.rename(columns={column: 'M_s'}, inplace=True) # star's mass (in solar masses)
            elif n%9 == 1:
                df.rename(columns={column: 'R_s'}, inplace=True) # star's radius (in solar radii)
            elif n%9 == 2:
                df.rename(columns={column: 'c_1'}, inplace=True) # limb darkening 1
            elif n%9 == 3:
                df.rename(columns={column: 'c_2'}, inplace=True) # limb darkening 2 
            else:
                break
        
        if n%9 == 0:
            df.rename(columns={column: f'planet'}, inplace=True) # indicates which member of the system it is
        elif n%9 == 1:
            df.rename(columns={column: f'Period_days'}, inplace=True) # planet's period (in days)
        elif n%9 == 2:
            df.rename(columns={column: f'T_0'}, inplace=True) # time of transit 0 (days in the BKJD system)
        elif n%9 == 3:
            df.rename(columns={column: f'sqrt(e) cos(omega)'}, inplace=True)
        elif n%9 == 4:
            df.rename(columns={column: f'sqrt(e) sin(omega)'}, inplace=True)          
        elif n%9 == 5:
            df.rename(columns={column: f'i'}, inplace=True) # inclination (degrees)
        elif n%9 == 6:
            df.rename(columns={column: f'Omega'}, inplace=True) # longitudiny of the mean node (default 0 degrees)
        elif n%9 == 7:
            df.rename(columns={column: f'M_pJ'}, inplace=True) # planet's mass (in Jupiter masses)
        elif n%9 == 8:
            df.rename(columns={column: f'R_p/R_s'}, inplace=True) # planet's radius / star's radius
        else:
            raise ValueError("how did we get here?")
            
    return df


# take the df and calculated parameters to it for more convenient analysis
def calculate_params(df):

    ## masses, radii , and densities
    df['R_pJ'] = df['R_s'] * df['R_p/R_s'] * (1/RJTORS) # radius of planet (Jupiter radii)
    df['R_pE'] = df['R_pJ']*RJTORE # radius of planet (Earth radii)
    df['M_pE'] = df['M_pJ'] / METOMJ # mass of planet (Earth masses)
    df['rho_p'] = df['M_pE'] * MEG / ((4/3) * np.pi * (df['R_pE']*RECM)**3) # planetary density (g/cm^3)
    df['rho_s'] = (df['M_s']*MSKG*1000) / ((4/3) * np.pi * (df['R_s']*RSCM)**3) # stellar density (g/cm^3)
    df['M_p/M_s'] = df['M_pE'] / (df['M_s']/METOMS) # mass of planet / mass of star
    
    ## orbital angles
    df['Omega'] = df['Omega'] % 360
    df['e'] = df['sqrt(e) cos(omega)']**2 + df['sqrt(e) sin(omega)']**2 # eccentricity
    df['omega'] = (np.arctan2(df['sqrt(e) sin(omega)'], df['sqrt(e) cos(omega)']) * 180/np.pi) % 360 # argument of periapse (degrees)
    df['true_anomaly'] = (90 - df['omega']) % 360
    df['eccentric_anomaly'] = ((180 / np.pi) * np.arctan2((np.sqrt(1-df['e']**2)*np.sin(df['true_anomaly']*np.pi/180)),(df['e']+np.cos(df['true_anomaly']*np.pi/180)))) % 360
    df['mean_anomaly'] = ((180 / np.pi) * ((np.pi / 180 ) * df['eccentric_anomaly']) - (df['e']*np.sin(df['eccentric_anomaly']*np.pi/180))) % 360 # M, the mean anomaly (19 degrees for KOI 500.01)
    df['mean_longitude'] = (df['Omega'] + df['omega'] + df['mean_anomaly']) % 360 # mean longitude of planet at epoch ::: longitude of ascending node (always 0 for our system) + argument of periapse (little omega) + mean anomaly (always close to 90 degrees)

    ## orbital distances
    df['a_AU'] = ((df['Period_days']*DTOS)**2 * G * ((df['M_s']*MSKG) + (df['M_pE']*MEKG))/(4*np.pi**2))**(1/3) * MTOAU # semimajor axis in AU
    df['a_R_s'] = (df['a_AU']/RSAU) / df['R_s'] # semimajor axis in stellar radii
    df['peri_AU'] = df['a_AU'] * (1 - df['e']) # periastron in AU
    df['peri_R_s'] = (df['peri_AU']/RSAU) / df['R_s'] # periastron in stellar radii
    df['apo_AU'] = df['a_AU'] * (1 + df['e']) # apoastron in AU
    df['apo_R_s'] = (df['apo_AU']/RSAU) / df['R_s'] # apoastron in stellar radii
    df['d_AU'] = df['a_AU']*(1 - df['e']**2) / (1 + (df['e']*np.cos(df['true_anomaly']*np.pi/180))) # star-planet separation at transit in AU
    df['d_R_s'] = (df['d_AU']/RSAU) / df['R_s'] # star-planet separation at transit in stellar radii

    ## impact, probability, and duration parameters DOUBLE CHECK TRANSIT IMPACT
    df['b_trans'] = (df['a_R_s'] * np.cos(df['i']*np.pi/180)) * ((1-df['e']**2)/(1+df['e']*np.sin(df['omega']*np.pi/180)))  # transit impact parameter
    df['b_occ'] = (df['a_R_s'] * np.cos(df['i']*np.pi/180)) * ((1-df['e']**2)/(1-df['e']*np.sin(df['omega']*np.pi/180))) # occultation impact parameter
    df['p_trans'] = ((df['R_s'] * RSAU + df['R_pJ']*RJAU) / df['a_AU']) * ((1+df['e']*np.sin(df['omega']*np.pi/180)) / (1-df['e']**2)) # transit probability
    df['p_occ'] = ((df['R_s'] * RSAU + df['R_pJ']*RJAU) / df['a_AU']) * ((1-df['e']*np.sin(df['omega']*np.pi/180)) / (1-df['e']**2)) # occultation probability
    df['T_total_hr'] = 24 * (df['Period_days'] / np.pi) * np.arcsin((df['R_s']*RSAU/df['a_AU'])*(np.sqrt((1+ df['R_p/R_s'])**2 - df['b_trans']**2)/np.sin(df['i']*np.pi/180))) * ((np.sqrt(1-df['e']**2))/(1+df['e']*np.sin(df['omega']*np.pi/180))) # total duration of transit (t4 - t1)
    df['T_full_hr'] = 24 * (df['Period_days'] / np.pi) * np.arcsin((df['R_s']*RSAU/df['a_AU'])*(np.sqrt((1-df['R_p/R_s'])**2 - df['b_trans']**2)/np.sin(df['i']*np.pi/180))) * ((np.sqrt(1-df['e']**2))/(1+df['e']*np.sin(df['omega']*np.pi/180))) # full duration of transit (t3 - t2)
    df['K_RV'] = (2*np.pi*G/(df['Period_days']*24*60*60))**(1/3) * ((MSKG*df['M_pJ']*np.sin(df['i']*np.pi/180)/MSTOMJ)/((df['M_s']*MSKG)+(MSKG*df['M_pJ']/MSTOMJ))**(2/3)) * (1/(1-df['e']**2)**(1/2))  # amplitude of radial velocity variations    ## make sure units are right here. should be m/s
    
    df = occurrence_rate_params(df)
    return df

#### should attach a flag for if the planet system is in the hsu et al paper
# attaches the occurrence rate parameters from hsu et al. 
def occurrence_rate_params(df):
    ocdf = pd.read_csv("occurrence_rates_hsu.csv")
    df["occurrence_rate_hsu"] = 0.0
    df["+sigma_hsu"] = 0.0
    df["-sigma_hsu"] = 0.0

    for i in df.index:
        mask = ((ocdf["radius_lower"] <= df.at[i,"R_pE"]) & 
                (ocdf["radius_upper"] > df.at[i,"R_pE"]) & 
                (ocdf["period_lower"] <= df.at[i,"Period_days"]) & 
                (ocdf["period_upper"] > df.at[i,"Period_days"]))
                
        df.at[i,"occurrence_rate_hsu"] = ocdf.loc[mask]["occurrence"].iloc[0]
        df.at[i,"+sigma_hsu"] = ocdf.loc[mask]["+sigma"].iloc[0]
        df.at[i,"-sigma_hsu"] = ocdf.loc[mask]["-sigma"].iloc[0]
        
    return df


# attaches the rowe table
def rowe_table_attach(koi,df):
    rowe_df = pd.read_csv("rowe_table_final.csv")
        
    num_new_cols = len(rowe_df.columns)

    new_cols = [f"new_col_{i}" for i in range(num_new_cols)]
    df[new_cols] = pd.DataFrame([[None]*num_new_cols], index=df.index)

    df.columns = list(df.columns[:len(df.columns)-num_new_cols]) + list(rowe_df.columns)
    row_rowe_match = rowe_df.loc[(float(koi) == round(rowe_df["KOI"]))]        
    
    for planet in df["planet"].unique():
    

        mask = (
            (row_rowe_match["Period_days_rowe"] > np.mean(df[df["planet"] == planet]["Period_days"]) - 0.2) &
            (row_rowe_match["Period_days_rowe"] < np.mean(df[df["planet"] == planet]["Period_days"]) + 0.2)
        )

        if not mask.any(): 
            print(f"Removing planet {planet} from df because no matches were found.")
            df = df[df["planet"] != planet]  
            with open("removed_planets_log.txt", "a") as file:
                file.write("koi: " + koi +" planet: "+ str(planet) +"\n")
                continue

        print("row_rowe_match.loc[mask, 'KIC':'e_BZ*'].values:", row_rowe_match.loc[mask, "KIC":"e_BZ*_rowe"].values)

        df.loc[df["planet"] == planet, "KIC":"e_BZ*_rowe"] = row_rowe_match.loc[mask].loc[:,"KIC":"e_BZ*_rowe"].values
        
    return df
    

# calculate parameters related to each planet's ordering in its system
def system_params(df):

    multiplicity = df['planet'].nunique()
    df['multiplicity'] = multiplicity  # number of planets per system
    
    # preset all comparisons to -1 (flag that the comparison is invalid, e.g. innermost planet / inner planet)
    df['P/Pin'] = -1.0
    df['P/Pout'] = -1.0
    df['Tdur/Tdurin'] = -1.0
    df['Tdur/Tdurout'] = -1.0
    df['R/Rin'] =  -1.0
    df['R/Rout'] = -1.0
    df['M/Min'] = -1.0
    df['M/Mout'] = -1.0
    df['rho/rhoin'] = -1.0
    df['rho/rhoout'] = -1.0
    df['i-iin'] = -1.0 
    df['iout-i'] = -1.0
    df['xiin'] = -1.0
    df['xiout'] = -1.0
    df['distin_hillrad'] = -1.0 
    df['distout_hillrad'] = -1.0 
    df['distin_hillrad_e'] = -1.0
    df['distout_hillrad_e']= -1.0 
    df['e/ein'] = -1.0
    df['eout/e'] = -1.0
    df['omega-omegain'] = -1.0
    df['omegaout-omega'] = -1.0

    # for each step of PhoDymm, there will be multiple planets
    for i in df.index.unique():
        pair_df = df.loc[i]
        # for each set of multiple planets, we will need to make comparisons between them
        for index, row in pair_df.iterrows():

            # pre-find some values for convenience
            planet=row["planet"]  
            period=row["Period_days"]
            t_dur=row["T_total_hr"]
            radius=row["R_pE"]
            mass=row["M_pE"]
            density=row["rho_p"]
            inc=row["i"]
            sm_axis=row["a_AU"]
            omega=row["omega"]
            ecc=row["e"]
            planet_star_mass_ratio=row["M_p/M_s"]

            # ordering boolean checks
            is_first = planet == 0.1
            is_last = planet*10 == multiplicity         
           
            # case: isn't the first planet
            if not is_first:
                
                df["P/Pin"].loc[(df.index == i) & (df["planet"]==planet)] = period / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "Period_days"].iloc[0]
                df['Tdur/Tdurin'].loc[(df.index == i) & (df["planet"]==planet)] = t_dur / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "T_total_hr"].iloc[0]
                df['R/Rin'].loc[(df.index == i) & (df["planet"]==planet)] = radius / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "R_pE"].iloc[0]
                df['M/Min'].loc[(df.index == i) & (df["planet"]==planet)] = mass / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "M_pE"].iloc[0]
                df['rho/rhoin'].loc[(df.index == i) & (df["planet"]==planet)] = density / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "rho_p"].iloc[0]
                df['i-iin'].loc[(df.index == i) & (df["planet"]==planet)] = (inc - pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "i"].iloc[0]) % 360
                df['omega-omegain'].loc[(df.index == i) & (df["planet"]==planet)] = (omega - pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "omega"].iloc[0]) % 360
                df['e/ein'].loc[(df.index == i) & (df["planet"]==planet)] = ecc / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "e"].iloc[0]
                df['xiin'].loc[(df.index == i) & (df["planet"]==planet)] = (pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "T_total_hr"].iloc[0] / t_dur) * (period / pair_df.loc[np.isclose(pair_df["planet"],planet-0.1), "Period_days"].iloc[0])**(1/3)
                df['distin_hillrad'].loc[(df.index == i) & (df["planet"]==planet)] = (sm_axis - pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"a_AU"].iloc[0]) / (((planet_star_mass_ratio+pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"M_p/M_s"].iloc[0])/3)**(1/3) * ((sm_axis + pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"a_AU"].iloc[0])/2))
                df['distin_hillrad_e'].loc[(df.index == i) & (df["planet"]==planet)] = ((sm_axis * (1 - ecc)) - (pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"a_AU"].iloc[0] * (1 + pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"e"].iloc[0]))) / (((planet_star_mass_ratio+pair_df.loc[np.isclose(pair_df["planet"],planet-0.1) ,"M_p/M_s"].iloc[0])/3)**(1/3) * ((sm_axis + pair_df.loc[np.isclose(pair_df["planet"] , planet-0.1) ,"a_AU"].iloc[0])/2))
            
            # case: isn't the last planet
            if not is_last:
                df["P/Pout"].loc[(df.index == i) & (df["planet"]==planet)] = period / pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "Period_days"].iloc[0]
                df['Tdur/Tdurout'].loc[(df.index == i) & (df["planet"]==planet)] = t_dur / pair_df.loc[np.isclose(pair_df["planet"],planet+0.1), "T_total_hr"].iloc[0]
                df['R/Rout'].loc[(df.index == i) & (df["planet"]==planet)] = radius / pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "R_pE"].iloc[0]
                df['M/Mout'].loc[(df.index == i) & (df["planet"]==planet)] = mass / pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "M_pE"].iloc[0]
                df['rho/rhoout'].loc[(df.index == i) & (df["planet"]==planet)] = density / pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "rho_p"].iloc[0]
                df['iout-i'].loc[(df.index == i) & (df["planet"]==planet)] = (pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "i"].iloc[0] - inc) % 360
                df['omegaout-omega'].loc[(df.index == i) & (df["planet"]==planet)] = (pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "omega"].iloc[0] - omega) % 360
                df['eout/e'].loc[(df.index == i) & (df["planet"]==planet)] = pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "e"].iloc[0] / ecc
                df['xiout'].loc[(df.index == i) & (df["planet"]==planet)] =  (t_dur / pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "T_total_hr"].iloc[0]) * (pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1), "Period_days"].iloc[0] / period)**(1/3)
                df['distout_hillrad'].loc[(df.index == i) & (df["planet"]==planet)] = (pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"a_AU"].iloc[0] - sm_axis) / (((planet_star_mass_ratio+pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"M_p/M_s"].iloc[0])/3)**(1/3) * ((sm_axis + pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"a_AU"].iloc[0])/2))
                df['distout_hillrad_e'].loc[(df.index == i) & (df["planet"]==planet)] = ((pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"a_AU"].iloc[0] * (1 - pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"e"].iloc[0])) - (sm_axis * (1 + ecc))) / (((planet_star_mass_ratio+pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"M_p/M_s"].iloc[0])/3)**(1/3) * ((sm_axis + pair_df.loc[np.isclose(pair_df["planet"] , planet+0.1) ,"a_AU"].iloc[0])/2))
         
    return df   

# creates an initial df from the list of subsampled rows
def make_df_from_subsample(subsampled_rows,koi):
    full_path = RAW_PATH + "koi" + koi + "/analysis_dir/dqa_allparam.csv"
    raw_data_df = pd.read_csv(full_path)
    return raw_data_df.loc[raw_data_df.index.isin(subsampled_rows["Unnamed: 0"])]
    
    
# tries to read in an output file from PhoDymm; if anything errors for processing that koi, this logs it 
def read_in_rows_write(breakpoints=False):
    for file in (os.listdir(SUBSAMPLED_PATH)):
        koi = find_koi(file)

        try:
            read_in_one_koi(koi)
            if breakpoints : 
                input()
                
        except Exception as e:
            print("koi ",koi," failed")
            with open("subsampler_error_log.txt", "a") as file:
                file.write(koi+"\n")
                
                
# reads in a single PhoDymm output for a given koi
def read_in_one_koi(koi):
    print(f"starting koi: {koi}")
    subsampled_rows = pd.read_csv(SUBSAMPLED_PATH+koi+'_subsamples.csv')
    single_df_write(subsampled_rows,koi)
    print(f"finished koi: {koi}")
        

# creates a df from the output file, processes it, and then writes it to the all_thin csv
def single_df_write(subsampled_rows,koi):
        df = make_df_from_subsample(subsampled_rows,koi)
        df = process_dataframe(df,koi)       
        write_header = not os.path.exists('thinned/all_thin.csv')
        df.to_csv('thinned/all_thin.csv', mode='a',header=write_header)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        read_in_rows_write()
    elif len(sys.argv) == 2:
        if find_koi(sys.argv[1]) is not None:
            read_in_one_koi(find_koi(sys.argv[1]))
    elif len(sys.argv) > 2:
        read_in_rows_write(breakpoints=True)
    else:
        print("invalid input")            