import numpy as np
from kg_priors import get_initial_guess_from_priors, prior_args

def get_initial_guess(nwalkers,ndim,model_id,method="priors"):
    if method == "priors":
        if model_id >= 1:
            p0 = np.zeros((nwalkers, ndim))  # Initialize an array for the initial guess
            assert ndim == len(prior_args.keys), "Number of dimensions must match the number of prior parameters!"
            for parameter_name, i in zip(prior_args.keys,range(ndim)):
                p0[:,i] = get_initial_guess_from_priors(parameter_name, nwalkers)  # Fill each column with initial guesses from priors
    else:
        raise ValueError("Unknown method for initial guess. Use 'priors'.")

    return p0    # This function should return an initial guess for the parametric model parameters.