import numpy as np
from scipy.special import gamma

def likelihood(params,observed,N_HSU_STARS,observation_probability):
    R_mrp = params[0]
    if observed < 0 : print("warning! observed is:",observed)
    if R_mrp < 0:
        return -np.inf
    return R_mrp * N_HSU_STARS * observation_probability * np.log(observed) - observed - np.log(gamma(R_mrp * N_HSU_STARS * observation_probability+1)) # test that this is maxed when expected == observed