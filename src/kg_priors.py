import numpy as np

# Format for prior arguments:
# 'parameter_name': (mu, sigma, type)  should try using parameters.csv instead?
#                   (lower, upper, type) for uniform distribution

class PriorArgs:
    def __init__(self):
        self.priors = []
        self.plot_labels = []
    def add_prior(self, parameter_name, mu, sigma, prior_type,model_id_list):
        self.priors.append([parameter_name,mu,sigma,prior_type,model_id_list])
    def get_priors(self,model_id):
        # Non-mutating filter. The old version removed non-matching entries
        # from self.priors while iterating over that same list -- a classic
        # mutate-during-iteration bug that silently skips the item right
        # after every removed one (list.remove shifts everything back one
        # slot, but the for loop's index still advances by one), AND
        # permanently shrinks self.priors, so a second call (e.g. the next
        # MCMC step, since kg_likelihood.py calls this once per step) filters
        # an already-filtered list and shrinks it further still. Returning a
        # fresh filtered list each time, without touching self.priors, fixes
        # both problems.
        return [prior for prior in self.priors if model_id in prior[4]]
    def get_prior_arguments(self,parameter_name):
        for prior in self.priors:
            if prior[0] == parameter_name:
                return prior[1], prior[2], prior[3]
        return None
    def get_initial_guess_from_priors(self, parameter_name, nwalkers):
        mu, sigma, prior_type =  self.get_prior_arguments(parameter_name)
        match prior_type:
            case "lnN":
                return np.random.lognormal(mu, sigma, nwalkers)
            case "N":
                return np.random.normal(mu, sigma, nwalkers)
            case "U":
                return np.random.uniform(mu,sigma,nwalkers) # for uniform, mu=lower, sigma=upper
            case _:
                raise ValueError(f"Unknown prior type: {prior_type}")
    def load_priors(self):
        # Gamma_0 (overall occurrence-rate normalization) is no longer a free
        # MCMC parameter. It has a closed-form conditional optimum given the
        # shape parameters (Gamma0_opt = n_planets / Lambda_tilde), and with
        # this prior's original uniform-in-log10(Gamma0) form, the exact
        # conditional posterior is Gamma0 | shape ~ Gamma(n_planets,
        # Lambda_tilde) -- see kg_likelihood.parametric_log_likelihood_pointprocess
        # and kg_plots.pointprocess_gamma0_posterior_plot, which reconstructs
        # Gamma0's full posterior after the fact from the per-step
        # lambda_tilde blob, so nothing about Gamma0 is actually lost by
        # dropping it from here.
        self.add_prior('gamma_0', -1,1,"U", [0,1])
        self.add_prior('gamma_1', -3, 5,"U", [0,1])  # lnN(0.6,0.1)
        self.add_prior('gamma_2', -1, 5,"U", [0,1])  # lnN(0,0.1)
        self.add_prior('sigma_0', 0, 5,"U", [0,1])  # lnN(-1.8, 0.25)
        self.add_prior('sigma_1', 0, 5,"U", [0,1])  # lnN(-1.3, 0.25)
        self.add_prior('sigma_2', 0, 5,"U", [0,1])  # lnN(-2.3, 0.25)
        self.add_prior('Mbreak1', 0.1, 50,"U", [0,1])  # lnN(2,1)
        self.add_prior('Mbreak2', 50, 10000,"U", [0,1])  # lnN(5,0.25)
        self.add_prior('C', 0.2,4.5,"U", [0,1])       
        self.add_prior('mu_M', 0, 10,"U", [0,1])  # N(1,2) 
        self.add_prior('sigma_M', -10, 10,"U", [0,1])  # lnN(1,0.25)
        self.add_prior('Beta1', 0.0, 5.0,"U", [0,1])  # N(0.5,0.5)
        self.add_prior('Beta2', -3.0, 0.0,"U", [0,1])  # N(-0.5,0.5)
        self.add_prior('Pbreak1', 3.0, 15,"U", [0,1])   # lnN(2,1)
        self.add_prior('alpha_e', 0,2,"U", [0])
        self.add_prior('lambda_e', 0,50,"U", [0])
        self.add_prior('sigma_e',0,1,"U", [0])

        # model_id 1: 2-component Gamma mixture on eccentricity, parametrized
        # by (mean, shape) per component -- see
        # kg_probability_distributions.eccentricity_log_pdf_gamma_mixture for
        # the full reasoning. mu_e_1/mu_e_2 are kept ADJACENT and in this
        # order deliberately: kg_likelihood.parametric_log_prior enforces
        # mu_e_1 < mu_e_2 by comparing params[i] to params[i+1], exactly the
        # way it already enforces Mbreak1 < Mbreak2 -- that ordering is what
        # keeps the two components identifiable (component 1 = tight/low-e,
        # component 2 = broad/higher-e) instead of being able to swap labels
        # between MCMC steps. Do not reorder these five without updating
        # both that check and the params[14:19] unpacking in
        # joint_log_intrinsic_density's model_id == 1 branch.
        #
        # Bounds are deliberately NOT flat/uninformative: alpha's upper bound
        # (15) is the load-bearing piece here -- coefficient of variation for
        # a Gamma is 1/sqrt(alpha), so an unbounded alpha lets a component
        # collapse into an arbitrarily narrow spike (mean held fixed, beta
        # scaled to match) with nothing pushing back on it, the same
        # soft-max/log-mean-exp collapse already diagnosed for the old
        # Rayleigh+Exponential sigma_e. mu ranges are informed by this
        # project's own multis (very tight, near e~0) vs. singles (broad,
        # extending past e~0.8) split, and cross-checked for
        # order-of-magnitude sanity against Stevenson et al. 2025's fitted
        # single-Gamma values (alpha ~ 1.2-1.5, implied mean e ~ 0.19-0.26,
        # for a RV sample -- not copied directly, since that split used known
        # multiplicity labels rather than an unconditioned mixture).
        self.add_prior('mu_e_1', 0.001, 0.15, "U", [1])   # tight/low-e component mean
        self.add_prior('mu_e_2', 0.1, 0.7, "U", [1])      # broad/higher-e component mean
        self.add_prior('alpha_e_1', 0.5, 15, "U", [1])    # tight component shape/concentration
        self.add_prior('alpha_e_2', 0.5, 15, "U", [1])    # broad component shape/concentration
        self.add_prior('f', 0, 1,"U", [1])                # mixing weight on component 1



        # self.add_prior('Log10(m)', -8,8,"U", [0])  
        # self.add_prior('Log10(p_noise)',-10,-3,"U", [0])  


        return self
    def load_plot_labels(self):
        self.plot_labels = ['$γ_0$',
                            '$γ_1$', 
                            '$γ_2$',  
                            '$σ_0$',  
                            '$σ_1$',   
                            '$σ_2$',  
                            '$M_{break,1}$',  
                            '$M_{break,2}$',   
                            'C',
                            r'$μ_M$',  
                            r'$σ_M$',  
                            '$β_1$',
                            '$β_2$',  
                            # '$β_3$',
                            '$P_{break,1}$',   
                            # '$P_{break,2}$',
                            '$α_e$',
                            '$λ_e$',
                            '$σ_e$',
                            '$μ_{e,1}$',
                            '$μ_{e,2}$',
                            '$α_{e,1}$',
                            '$α_{e,2}$',
                            '$f$'
                            # r'$\mathrm{log}_{10}(m)$',
                            # r'$\mathrm{logit}(P_{noise})$'
                            ]
    def get_plot_labels(self, model_id):
        # Same non-mutating fix as get_priors above (mutate-during-iteration
        # bug, plus it was permanently shrinking both self.priors and
        # self.plot_labels on every call -- self.priors[i] no longer lines up
        # with self.plot_labels[i] the second time this or get_priors runs).
        self.load_plot_labels()
        return [label for prior, label in zip(self.priors, self.plot_labels) if model_id in prior[4]]
        

            