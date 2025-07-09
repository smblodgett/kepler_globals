import numpy as np

# Format for prior arguments:
# 'parameter_name': (mu, sigma, type)  should try using parameters.csv instead?
#                   lower, upper
prior_args = {
        'Gamma_0': (0, 1,"lnN"),  # lnN(0,1)
        'gamma_0': (0,0.1,"N"),
        'gamma_1': (0.6, 0.1,"lnN"),  # lnN(0.6,0.1)
        'gamma_2': (0, 0.1,"lnN"),  # lnN(0,0.1)
        'sigma_0': (-1.8, 0.25,"lnN"),  # lnN(-1.8, 0.25)
        'sigma_1': (-1.3, 0.25,"lnN"),  # lnN(-1.3, 0.25)
        'sigma_2': (-2.3, 0.25,"lnN"),  # lnN(-2.3, 0.25)
        'Mbreak1': (2, 1,"lnN"),  # lnN(2,1)
        'Mbreak2': (5, 0.25,"lnN"),  # lnN(5,0.25)
        'C': (2.5,1,"N"),
        'mu_M': (1, 2,"N"),  # N(1,2)
        'sigma_M': (1, 0.25,"lnN"),  # lnN(1,0.25)
        'Beta1': (0.5, 0.5,"N"),  # N(0.5,0.5)
        'Beta2': (-0.5, 0.5,"N"),  # N(-0.5,0.5)
        'Beta3': (-1.0,1.0,"N"),
        'Pbreak1': (2, 1,"lnN"),   # lnN(2,1)
        'Pbreak2':(5,1.5,"lnN"),
        'alpha_e': (0,1,"U"),
        'lambda_e': (3,2,"N"),
        'sigma_e':(0,1,"U")
        }


def get_prior_arguments(parameter_name):
    """Return the prior arguments for a given parameter."""
    return prior_args.get(parameter_name)


def get_initial_guess_from_priors(parameter_name, nwalkers):
    """Return an initial guess for a parameter based solely on its prior."""
    mu, sigma, prior_type =  get_prior_arguments(parameter_name)
    match prior_type:
        case "lnN":
            return np.random.lognormal(mu, sigma, nwalkers)
        case "N":
            return np.random.normal(mu, sigma, nwalkers)
        case "U":
            return np.random.uniform(mu,sigma,nwalkers)