import numpy as np
import pandas as pd
import json
from kg_grid_object_hook import grid_object_hook

global voxel_grid
with open('../data/voxel_grid.json', "r") as f:
    voxel_grid = json.load(f,object_hook=grid_object_hook)




def find_completeness(r,p,m,e,o):
    completeness = voxel_grid.interpolate_completeness(np.array([r,p,m,e,o]))
    return completeness
    