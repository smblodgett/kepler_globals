# main script that runs Kepler Globals modeling: as of now, this is modeling the voxels of the M-P-R relationship

# begun October 3, 2024
# developed by Steven Blodgett, with Darin Ragozzine, Dallin Spencer, and Daniel Jones
# codebase drawn from Dallin Spencer's multi_moon and David Jensen's gaiamr


import commentjson as json
import pandas as pd
import sys
import os
import datetime
import shutil
import numpy as np

from kg_griddefiner import *

# object to read in runprops file
class ReadJson:
    def __init__(self, filename):
        print('Read the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data
    
# verify path
cwd = os.getcwd()
print(cwd)        

# find runprops path
if 'src' in cwd:
    runprops_filename = "../runs/runprops.txt"
elif 'runs' in cwd:
    runprops_filename = "runprops.txt"
elif 'results' in cwd:
    runprops_filename = "runprops.txt"
else:
    print('You are not starting from a proper directory. You should run kg_run.py from a src, runs, or a results directory.')
    sys.exit()
    
# get runprops loaded in, find initial guess file
getData = ReadJson(runprops_filename)
runprops = getData.outProps()
init_filename = runprops.get("init_filename")

# get the other parameters from runprops
verbose = runprops.get("verbose")
object_name = runprops.get("object_name")
nwalkers = runprops.get("nwalkers")
nsteps = runprops.get("nsteps")
nburnin = runprops.get("nburnin")
make_plots = runprops.get("make_plots")
output_filename = runprops.get("output_filename")

voxel_grid = create_voxel_array(radius_grid_array,period_grid_array,mass_grid_array)


for voxel in voxel_grid.flatten():
    print(voxel)