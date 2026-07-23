import numpy as np
from kg_utilities import ReadJson
from kg_priors import PriorArgs

def get_initial_guess(nwalkers,ndim,model_id,method="priors",previous_filename="",custom_initial_guess=None):
    best_params = None  # Initialize best_params to None
    best_log_prob = None
    if method == "priors":
        p0 = np.zeros((nwalkers, ndim))  # Initialize an array for the initial guess
        prior_args = PriorArgs().load_priors()
        priors = prior_args.get_priors(model_id)
        assert ndim == len(priors), "Number of dimensions must match the number of prior parameters!"
        for prior, i in zip(priors, range(ndim)):
            parameter_name = prior[0]
            p0[:, i] = prior_args.get_initial_guess_from_priors(parameter_name, nwalkers)  # Fill each column with initial guesses from priors
        print("using priors initialization method")
    
    elif method == "previous_best":
        assert previous_filename is not None, "Enter the filename of the run you want to take!"

        p0 = np.zeros((nwalkers,ndim))

        best_params, best_log_prob = get_initial_guess_from_previous(previous_filename)
        assert len(best_params) == ndim, "Mismatch between loaded best params and expected ndim!"

        scale = 1e-2 * np.maximum(np.abs(best_params), 1e-1) # if any best param is zero, it needs to use a small fixed scale
        np.random.seed(42)
        p0 = np.random.normal(best_params,scale=scale,size=(nwalkers,len(best_params)))
        print("using previous best initialization method")
        
    elif method == "custom":
        assert custom_initial_guess is not None, "Enter the custom initial guess!"
        assert custom_initial_guess.shape == (ndim,), "Custom initial guess must have shape (ndim)!"
        p0 = np.zeros((nwalkers,ndim))
        # Fill p0 with custom initial guesses
        for i in range(ndim):
            p0[:, i] = np.random.normal(custom_initial_guess[i], scale=1e-2 * np.maximum(np.abs(custom_initial_guess[i]), 1e-1), size=nwalkers)
        print("using custom initial guess method")
    else:
        raise ValueError("Unknown method for initial guess. Use 'priors'.")

    return p0, best_params, best_log_prob    # This function should return an initial guess for the parametric model parameters.
 

def get_initial_guess_from_previous(filename):
    previous_best = ReadJson(filename).outProps()
    print("previous_best['params']: ", previous_best["params"])
    print("type(previous_best['params']): ",type(previous_best["params"]))
    # previous_best_likelihood = previous_best["log_prob"]
    print("previous best: ",previous_best["params"])

    return np.array(previous_best["params"]), previous_best["log_prob"]

