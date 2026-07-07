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
        for prior in self.priors:
            if model_id in prior[4]:
                continue
            else:
                self.priors.remove(prior)
        return self.priors
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
        self.add_prior('Log10(Gamma_0)', -4, 2.0,"U", [0,1])  # now log10(Gamma0)
        self.add_prior('gamma_0', -1,1,"U", [0,1])
        self.add_prior('gamma_1', -1.5, 1.5,"U", [0,1])  # lnN(0.6,0.1)
        self.add_prior('gamma_2', -1, 2,"U", [0,1])  # lnN(0,0.1)
        self.add_prior('sigma_0', 0, 2,"U", [0,1])  # lnN(-1.8, 0.25)
        self.add_prior('sigma_1', 0, 2,"U", [0,1])  # lnN(-1.3, 0.25)
        self.add_prior('sigma_2', 0, 2,"U", [0,1])  # lnN(-2.3, 0.25)
        self.add_prior('Mbreak1', 0.1, 50,"U", [0,1])  # lnN(2,1)
        self.add_prior('Mbreak2', 50, 10000,"U", [0,1])  # lnN(5,0.25)
        self.add_prior('C', 0.2,4.5,"U", [0,1])       
        self.add_prior('mu_M', 0, 10,"U", [0,1])  # N(1,2) 
        self.add_prior('sigma_M', -10, 10,"U", [0,1])  # lnN(1,0.25)
        self.add_prior('Beta1', 0.0, 5.0,"U", [0,1])  # N(0.5,0.5)
        self.add_prior('Beta2', -5.0, 5.0,"U", [0,1])  # N(-0.5,0.5)
        self.add_prior('Pbreak1', 0.0, 20,"U", [0,1])   # lnN(2,1)
        self.add_prior('alpha_e', 0,2,"U", [0,1])
        self.add_prior('lambda_e', 0,50,"U", [0,1])
        self.add_prior('sigma_e',0,1,"U", [0,1])
        # self.add_prior('Log10(m)', -8,8,"U", [0])  
        # self.add_prior('Logit(p_noise)',-10,10,"U", [0])  


        return self
    def load_plot_labels(self):
        self.plot_labels = [r'$\mathrm{log}_{10}(Γ_0)$',
                            '$γ_0$',
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
                            r'$\mathrm{log}_{10}(m)$',
                            r'$\mathrm{logit}(P_{noise})$'
                            ]
    def get_plot_labels(self, model_id):
        self.load_plot_labels()
        for prior, prior_label in zip(self.priors, self.plot_labels):
            if model_id in prior[4]:
                continue
            else:
                self.priors.remove(prior)
                self.plot_labels.remove(prior_label)
        return self.plot_labels
        

            