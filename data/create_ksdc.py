import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.csv as ar_csv

from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent / "src"))
from kg_initialize_voxel_grid import process_singles_df


stellar_data_filename = "../data/berger_2020_keplerstellar.tsv"
rowe_stellar_data_filename ="../data/rowe_table_final.csv"
dr_25_data_filename = "../data/q1_q17_dr25.csv"


stellar_df = pd.read_csv(stellar_data_filename,engine='pyarrow',delimiter='\t') # used to be from ../data/keplerstellar.csv, now is from Berger et al 2020

# Read in the expanded stellar df, which has CDPP values
rowe_stellar_df = pd.read_csv(rowe_stellar_data_filename,engine='pyarrow') # this is the stellar data from Rowe et al 2015.
# rowe_stellar_df = rowe_stellar_df[rowe_stellar_df["st_delivname"]=="q1_q17_dr25_stellar"]

print("len(stellar_df) before cuts: ",len(stellar_df))

# Make the cuts to stellar catalog based off of temperature, logg
stellar_df = stellar_df[(stellar_df["Teff"]>4000) & (stellar_df["Teff"]<7000)]
stellar_df = stellar_df[(stellar_df["logg"]>4)]

print("len(stellar_df) after hsu cuts: ",len(stellar_df))

stellar_df = stellar_df.merge(
                            rowe_stellar_df,
                            left_on='KIC',
                            right_on='KIC',
                            how='left'   # keeps all rows from stellar_df
                        )

dr_df = pd.read_csv(dr_25_data_filename,engine='pyarrow')

# create a column for the multipliity of each DR25 planet
dr_df['multiplicity'] = dr_df['kepid'].map(dr_df['kepid'].value_counts())


singles_dr_df = dr_df[dr_df["multiplicity"]==1]  

print("sum singles_dr_df['kepid'].isin(stellar_df['KIC']) : ", np.sum(singles_dr_df["kepid"].isin(stellar_df['KIC'])))

singles_dr_df = singles_dr_df[singles_dr_df["kepid"].isin(stellar_df['KIC'])]



# Remove the planets in the singles df that have nans in their period errors, since we need these for sampling the posteriors
singles_dr_df = singles_dr_df[~(singles_dr_df["koi_period_err1"].isna() | singles_dr_df["koi_period_err2"].isna())]
# Reset the index so we can iterate through singles df
singles_dr_df = singles_dr_df.reset_index(drop=True)
# Give the singles df the same cols as the multis df, sample ecc and omega for the singles
processed_singles_dr_df = process_singles_df(singles_dr_df,stellar_df,0.01,10,seed=333,validation_graph=False,make_graphs=False)

print("finished processing!")

processed_singles_dr_df = processed_singles_dr_df.merge(
                                                        stellar_df,
                                                        left_on='kepid',
                                                        right_on='KIC',
                                                        how='left'
                                                )

print("finished merging!")


table = pa.Table.from_pandas(processed_singles_dr_df)
ar_csv.write_csv(table, f"ksdc_7_30.csv")


print(f"Saved ksdc")
