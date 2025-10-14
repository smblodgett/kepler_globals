import numpy as np

# Format for prior arguments:
# 'parameter_name': (mu, sigma, type)  should try using parameters.csv instead?
#                   (lower, upper, type) for uniform distribution
prior_args = {
        'Gamma_0': (-4, 1.2,"U"),  # now log10(Gamma0)
        'gamma_0': (-1,1,"U"),
        'gamma_1': (0, 1.5,"U"),  # lnN(0.6,0.1)
        'gamma_2': (-1, 1,"U"),  # lnN(0,0.1)
        'sigma_0': (0, .5,"U"),  # lnN(-1.8, 0.25)
        'sigma_1': (0, .5,"U"),  # lnN(-1.3, 0.25)
        'sigma_2': (0, .5,"U"),  # lnN(-2.3, 0.25)
        'Mbreak1': (0.1, 50,"U"),  # lnN(2,1)
        'Mbreak2': (50, 10000,"U"),  # lnN(5,0.25)
        'C': (1.5,3.5,"U"),
        'mu_M': (-.5, 3,"U"),  # N(1,2)
        'sigma_M': (.5, 3.5,"U"),  # lnN(1,0.25)
        'Beta1': (.5, 1.5,"U"),  # N(0.5,0.5)
        'Beta2': (-1.5, -.5,"U"),  # N(-0.5,0.5)
        'Beta3': (-2.5,2.5,"U"),
        'Pbreak1': (0.2, 20,"U"),   # lnN(2,1)
        'Pbreak2':(20,500,"U"),
        'alpha_e': (0,1,"U"),
        'lambda_e': (0,5,"U"),
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
        
        
def apply_priors_to_params(params):
    
    return params