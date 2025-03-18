'''
January 29, 2025
@author: Steven Blodgett
Generate a list, N_SAMPLE in length, of rows to use from a given PhoDyMM dqa final file. 
The first 2% of rows are ignored. Rows are sampled without replacement. 
Remove every chain after the first chain with a rejection_constant of 50 or more (max 25 can be removed).

'''


import pandas as pd
import numpy as np
import os
import sys
import re

PHODYMM_PATH = "/hdd2/backup/danielkj/PhoDyMM_results_final/completed_systems"
SAVE_PATH = '/home/byu.local/smb9564/research/hierarchal_modeling/kepler_globals/data/subsampled_rows/' 
N_SAMPLE = 1000

def find_dqas():
    for folder in os.listdir(PHODYMM_PATH):
#         if len(dqas) > 2:
#             break
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
  

# grab the koi number from filename
def find_koi(name_with_numbers):
    numbers = re.findall(r'\d+', name_with_numbers)
    result = ''.join(numbers)
    return result[1:]


def random_select(koi,df):
         
    filtered_df = pd.DataFrame()

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


# attaches the rejection constant, a measure of how anomalous a given chain is.
def clustering_rejection(df):
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

# implements algorithm to assign each walker the rejection constant
def accept_reject_algorithm(sorted_dict):
    accept_reject_constants = []

    for const in np.arange(2, 1001):
        accept_reject_chain = []
        
        values = list(sorted_dict.values())  # precompute list for indexing
        
        for i, (chain, likelihood) in enumerate(sorted_dict.items()):
            
            if i == 0:  # first element will always pass
                accept_reject_chain.append([1, const, chain])
                continue
            
            if i == len(sorted_dict) - 1:
#                 print("i=",i)
#                 print("likelihood",likelihood)
#                 print("chain=",chain)
                next_jump = values[i] - values[i-1]  # give the final chain the value of the previous chain 
#                 print("next jump=",next_jump)
#                 print("constant=",const)
#                 print("initial to now jump=",const * (values[i] - values[0]) / (i))
#                 input()

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
    
    chain_means=[]
    for chain in filtered_df["Chain#"].unique():
        chain_df = filtered_df[filtered_df["Chain#"]==chain]
        chain_mean = np.mean(chain_df['chisq'])
        chain_means.append([chain,chain_mean,np.mean(chain_df['rejection_const'])])
    chain_means_sorted = sorted(chain_means, key=lambda x: x[1])
    
    cutoff_reached=False
    num_removed=0
    for means in chain_means_sorted:
        chain = means[0]
        chain_mean = means[1]
        rejection_constant = means[2]
        if cutoff_reached==True:
            num_removed+=1
            continue
        if rejection_constant >= cutoff:
            cutoff_reached=True
            num_removed+=1
            print("hit cutoff")
    
    if num_removed>25:
        num_removed=25
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
