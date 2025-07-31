import numpy as np
from kg_utilities import ReadJson
from kg_priors import get_initial_guess_from_priors, prior_args

def get_initial_guess(nwalkers,ndim,model_id,method="priors",previous_filename=""):
    if method == "priors":
        p0 = np.zeros((nwalkers, ndim))  # Initialize an array for the initial guess
        assert ndim == len(prior_args.keys()), "Number of dimensions must match the number of prior parameters!"
        for parameter_name, i in zip(prior_args.keys(),range(ndim)):
            p0[:,i] = get_initial_guess_from_priors(parameter_name, nwalkers)  # Fill each column with initial guesses from priors
        print("using priors initialization method")
    elif method == "previous_best":
        assert previous_filename is not None, "Enter the filename of the run you want to take!"

        p0 = np.zeros((nwalkers,ndim))

        best_params = get_initial_guess_from_previous(previous_filename)
        assert len(best_params) == ndim, "Mismatch between loaded best params and expected ndim!"

        scale = 1e-5 * np.abs(best_params)
        p0 = np.random.normal(best_params,scale=scale,size=(nwalkers,len(best_params)))
        print("using previous best initialization method")
    else:
        raise ValueError("Unknown method for initial guess. Use 'priors'.")

    return p0    # This function should return an initial guess for the parametric model parameters.




def get_initial_guess_from_previous(filename):
    previous_best = ReadJson(filename).outProps()
    print("previous_best['params']: ", previous_best["params"])
    print("type(previous_best['params']): ",type(previous_best["params"]))

    return np.array(previous_best["params"])

