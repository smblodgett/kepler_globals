import numpy as np

# Format for prior arguments:
# 'parameter_name': (mu, sigma, type)  should try using parameters.csv instead?
prior_args = {
        'Gamma_0': (0, 1,"lnN"),  # lnN(0,1)
        'gamma_1': (0.6, 0.1,"lnN"),  # lnN(0.6,0.1)
        'gamma_2': (0, 0.1,"lnN"),  # lnN(0,0.1)
        'sigma_0': (-2.8, 0.25,"lnN"),  # lnN(-2.8, 0.25)
        'sigma_1': (-1.3, 0.25,"lnN"),  # lnN(-1.3, 0.25)
        'sigma_2': (-2.3, 0.25,"lnN"),  # lnN(-2.3, 0.25)
        'sigma_3': (-1.5, 0.25,"lnN"),  # lnN(-1.5, 0.25)
        'Mbreak1': (2, 1,"lnN"),  # lnN(2,1)
        'Mbreak2': (5, 0.25,"lnN"),  # lnN(5,0.25)
        'mu_M': (1, 2,"N"),  # N(1,2)
        'sigma_M': (1, 0.25,"lnN"),  # lnN(1,0.25)
        'Beta1': (0.5, 0.5,"N"),  # N(0.5,0.5)
        'Beta2': (-0.5, 0.5,"N"),  # N(-0.5,0.5)
        'Pbreak1': (2, 1,"lnN")   # lnN(2,1)
        }


def get_prior_arguments(parameter_name):
    """Return the prior arguments for a given parameter."""
    return prior_args.get(parameter_name)


def get_initial_guess_from_priors(parameter_name, nwalkers):
    """Return an initial guess for a parameter based solely on its prior."""
    mu, sigma, type =  get_prior_arguments(parameter_name)
    if type == "lnN":
        return np.random.lognormal(mu, sigma, nwalkers)
    elif type == "N":
        return np.random.normal(mu, sigma, nwalkers)