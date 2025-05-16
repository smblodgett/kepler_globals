import numpy as np
from scipy.special import gamma

def likelihood(params,observed,N_HSU_STARS,observation_probability):
    R_mrp = params[0]
    # sum from i=1 to n of (xi - log(parameter) - parameter - log(Gamma(xi+1)))
    # but I think this might only be n=1 total because we only have one...dimension?
    if observed < 0 : print("observed:",observed)
    
#     if gamma(expected+1) <0 : 
#         print(expected)
#         print("gamma(expected+1)",gamma(expected+1))
    if R_mrp < 0:
        return -np.inf
    return R_mrp * N_HSU_STARS * observation_probability * np.log(observed) - observed - np.log(gamma(R_mrp * N_HSU_STARS * observation_probability+1)) # test that this is maxed when expected == observed