'''
kg_random_row_selector.py
=========================

Randomly selects a list of rows for each koi file.

Generates a list, N_SAMPLE in length, of rows to use from a given PhoDyMM dqa final data file. 
The first 2% of rows are ignored. Rows are sampled without replacement. 
Removes every chain after the first chain with a rejection_constant of 50 or more (max 25 can be removed).
The rejection constant is described in Hou et al. 2012, equation 17, though this implementation
has been modified slightly. 

Usage
-----
Run from the command line:
    python kg_random_row_selector.py koi_number
    
Parameters
----------
koi_number : int, optional
    Specifies which koi system the random row selector will run on. If no 
    koi is specified, the selector will run on all available systems.

Author
------
Steven Blodgett <blodgett.steven.m@gmail.com>
Created on: 2025-01-29
'''


import pandas as pd
import numpy as np
import os
import sys
import re

PHODYMM_PATH = "/hdd2/backup/danielkj/PhoDyMM_results_final/completed_systems" # path to the PhoDyMM final results
SAVE_PATH = '/home/byu.local/smb9564/research/hierarchal_modeling/kepler_globals/data/subsampled_rows/' # path to the save location of the list of rows for each system
N_SAMPLE = 1000 # number of rows to select


def find_dqas():
    """Finds all PhoDyMM output dqa files and performs random selection on each one, saving chosen rows."""
    for folder in os.listdir(PHODYMM_PATH):
        folder_path = os.path.join(PHODYMM_PATH, folder)
        if os.path.isdir(folder_path):  
            for path in os.listdir(folder_path):
                path_full = os.path.join(folder_path, path)
                if os.path.isdir(path_full) and path == 'analysis_dir':
                    for file in os.listdir(path_full):
                        if file == 'dqa_allparam.csv':
                            path_full = os.path.join(path_full, file)
                            print(path_full)
                            koi = find_koi(path_full)
                            print(koi)
                            df = pd.read_csv(path_full)
                            random_select(koi,df)

                            
def find_koi(name_with_numbers):
    """Grabs the koi number from filename and returns it as a string."""
    numbers = re.findall(r'\d+', name_with_numbers)
    result = ''.join(numbers)
    return result[1:]


def random_select(koi,df):
    """
    Randomly select N_THINNED rows from a given koi's dataframe.
    
    Calls the clustering algorithm on the dataframe to determine which chains need to be 
    removed, then removes them from the dataframe, writing which chains are removed to 
    file "rejected_chains.txt". Then, randomly selects N_THINNED rows for the subsampled 
    data set (without replacement). Then, it writes to  SAVE_PATH/{koi}_subsamples.csv 
    the list of rows that are kept, along with the chain number and chisq values for 
    external validation purposes. 
    
    Parameters
    ----------
    df : DataFrame
      The dataframe derived from one PhoDyMM output posterior.
    koi : str
      The koi number of the system (there will often be a leading 0).
    
    Returns
    -------
    None
    """
    filtered_df = pd.DataFrame()

    # Sort by chain number. 
    grouped = df.groupby('Chain#')
    for chain, chain_df in grouped:
        chain_length = len(chain_df)
        removal_value = round(chain_length * 0.02)   
        filtered_chain_df = chain_df.iloc[removal_value:]
        filtered_df = pd.concat([filtered_df, filtered_chain_df], axis=0)

    filtered_df = clustering_rejection(filtered_df)      
    
    rejection_cutoff = 50
    
    filtered_df = rejection_constant_cutoff(filtered_df,rejection_cutoff)
    
    rejected_chains = filtered_df[filtered_df['rejection_const'] >= rejection_cutoff]["Chain#"].unique()
    
    filtered_df = filtered_df[filtered_df['rejection_const'] < rejection_cutoff]
    print(filtered_df)
    print(filtered_df[filtered_df['rejection_const'] > rejection_cutoff])
    print(rejected_chains)
    
    with open("rejected_chains.txt", "a") as f:
        f.write(f"{koi} "+str(rejected_chains)+" \n")
                   
    rows_to_keep = np.random.choice(filtered_df.index, 1000, replace=False)

    df_kept = filtered_df.loc[rows_to_keep]

    df_kept[['Unnamed: 0','Chain#','chisq']].to_csv(SAVE_PATH+koi+'_subsamples.csv')


def clustering_rejection(df):
    """Attaches the rejection constant as a column to a dataframe, a measure of how anomalous a given chain is."""
    default_final_rejection_constant = 2
    df['rejection_const'] = 0
    mean_likelihoods = {}
    df['Chain#'] = df['Chain#'].astype(int)
    print(max(df['Chain#']))
    for chain in range(int(max(df['Chain#']))+1):
        chain_df = df[df['Chain#']==chain]
        mean_likelihood = np.mean(chain_df['chisq'])
        mean_likelihoods[chain] = mean_likelihood
    
    sorted_dict = dict(sorted(mean_likelihoods.items(), key=lambda item: item[1]))
    
    accept_reject_constants = accept_reject_algorithm(sorted_dict)
    
    return add_rejection_constant_helper(df,accept_reject_constants,default_final_rejection_constant)


def accept_reject_algorithm(sorted_dict):
    """
    Implements algorithm to assign each walker the rejection constant. 
    
    Its return value is parsed by the add_rejection_constant_helper function to actually make 
    the assignment.
    
    Parameters
    ----------
    sorted_dict : dict
      A dictionary of form {chain number : chain likelihood} sorted by likelihood.
    
    Returns
    -------
    accept_reject_constants : list(list(list()))
      A nested list. The innermost list is of form [<flag>, clustering constant, chain number].
      It iterates over all of the sorted chain numbers/likelihoods. A flag value of 1 indicates 
      that this chain passes with this clustering constant, and 0 indicates that this chain does 
      not pass. The middle list iterates over the accept/reject constant values from 2 to 1001. 
      1001 is an additional flag; very few chains should reach this value; if many do, it means
      something is wrong in the code.
    """
    accept_reject_constants = []
    # Check each constant value from 2 to 1001. 
    for const in np.arange(2, 1001):
        accept_reject_chain = []
        
        values = list(sorted_dict.values())  # Precompute list for indexing.
        
        for i, (chain, likelihood) in enumerate(sorted_dict.items()):
            
            if i == 0:  # The first element will always pass.
                accept_reject_chain.append([1, const, chain])
                continue
            
            if i == len(sorted_dict) - 1:
                next_jump = values[i] - values[i-1]  # Give the final chain the value of the previous chain. 
            else:
                next_jump = values[i+1] - values[i]

            initial_to_now_jump = const * (values[i] - values[0]) / (i)

            if next_jump < initial_to_now_jump:
                accept_reject_chain.append([1, const, chain])
            else:
                accept_reject_chain.append([0, const, chain])
        
        accept_reject_constants.append(accept_reject_chain)
    
    return accept_reject_constants


# helper function for rejection constant attachment
def add_rejection_constant_helper(df,accept_reject_constants,default_rejection_constant):
    """
    A helper function for attaching the rejection constant to a dataframe.
    
    Parameters
    ----------
    df : DataFrame
      The dataframe derived from one PhoDyMM output posterior.
    accept_reject_constants : list(list(list()))
      A nested list. The innermost list is of form [<flag>, clustering constant, chain number].
      It iterates over all of the sorted chain numbers/likelihoods. A flag value of 1 indicates 
      that this chain passes with this clustering constant, and 0 indicates that this chain does 
      not pass. The middle list iterates over the accept/reject constant values from 2 to 1001. 
      1001 is an additional flag; very few chains should reach this value; if many do, it means
      something is wrong in the code.
    default_rejection_constant : int
      The default value assigned as the clustering constant if the value is NaN.
    
    Returns
    -------
    df : DataFrame
      The modified dataframe; each row should now have a rejection constant assigned to it. 
    """
    chain_rejection_dict = {}
    for constant_layer in accept_reject_constants:
        for k, terms in enumerate(constant_layer):
            if terms[0] == 1:
                if terms[2] not in chain_rejection_dict:
                    chain_rejection_dict[terms[2]] = terms[1]
                elif terms[2] in chain_rejection_dict:
                    continue
                else:
                    raise ValueError
            elif terms[0] == 0:
                continue
    
    df['rejection_const'] = df['Chain#'].map(chain_rejection_dict).fillna(default_rejection_constant)
    return df    


def rejection_constant_cutoff(filtered_df,cutoff):
    """
    Takes a dataframe and removes chains with a high clustering constant.
    
    Any chain with a clustering constant greater than cutoff is noted; then, any chain 
    with a lower mean likelihood than that chain is given a clustering constant equal to cutoff,
    starting with the worst. However, only 25 chains will be marked in this way total. 
    
    Parameters
    ----------
    filtered_df : DataFrame
      A PhoDyMM output dataframe which has been assigned a rejection constant.
    cutoff : int
      The value above which chains should be rejected (50 is recommended).
    
    Returns
    -------
    filtered_df : DataFrame
      The dataframe with all bad chains marked. 
    """
    # Sort the means of the likelihood of each chain.
    chain_means = []
    for chain in filtered_df["Chain#"].unique():
        chain_df = filtered_df[filtered_df["Chain#"]==chain]
        chain_mean = np.mean(chain_df['chisq'])
        chain_means.append([chain,chain_mean,np.mean(chain_df['rejection_const'])])
    chain_means_sorted = sorted(chain_means, key=lambda x: x[1])
    
    cutoff_reached = False
    num_removed = 0
    for means in chain_means_sorted:
        chain = means[0]
        chain_mean = means[1]
        rejection_constant = means[2]
        if cutoff_reached == True:
            num_removed += 1
            continue
        if rejection_constant >= cutoff:
            cutoff_reached = True
            num_removed += 1
            print("hit cutoff")
    
    if num_removed > 25:
        num_removed = 25
    if num_removed > 0:
        for i in range(len(chain_means_sorted)-num_removed-1,len(chain_means_sorted)):
            print(i)
            chain = chain_means_sorted[i][0]
            mean = chain_means_sorted[i][1]
            print("chain,mean=",chain,mean)
            filtered_df.loc[filtered_df['Chain#'] == chain, 'rejection_const'] = cutoff
    print("finished")
    return filtered_df


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        find_dqas()
    elif len(sys.argv) == 2:
        koi = find_koi(sys.argv[1])
        df = pd.read_csv(sys.argv[1])
        random_select(koi,df)
    else:
        print("invalid input")