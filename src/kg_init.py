import numpy as np
import pandas as pd
"""
This file will have the user input their initial guess for where to start in the emcee program
"""


def kg_init_guess(runprops):
    """This function will produce the initial guess used in multimoon.
    
    Input: 
    
    runprops- All run properties for the code. Will include
        the name of the init_guess dataframe csv file.
    
    Returns:    
    params_df - A parameters dataframe with the same column names
        as start_guess_df and nwalker rows drawn from the 
        distribution.
    """
    
    
    
    return params_df