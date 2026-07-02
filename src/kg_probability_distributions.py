import numpy as np
from numba import njit
import time
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.stats import lognorm # truncnorm #, gaussian_kde
# from scipy.stats import gamma as gamma_dist
from scipy.special import gamma, ndtr, ndtri


from kg_constants import G, RETORS, RSCM, MSKG, MEKG, RECM, RSCM
from kg_utilities import radius_given_density_mass
from kg_param_boundary_arrays import radius_grid_array, period_grid_array, mass_grid_array, eccentricity_grid_array, omega_grid_array


class PeriodDistribution:
    def __init__(self, period_fine_grid, betas, breaks, power_laws=3):
        self.period_fine_grid = period_fine_grid
        self.β1 = betas[0]
        if power_laws > 1:
            self.β2 = betas[1]
            self.Period_break_1 = breaks[0]
        if power_laws > 2:
            self.β3 = betas[2]
            self.Period_break_2 = breaks[1]

        self.power_laws = power_laws
        assert power_laws in [1, 2, 3], "power_laws must be 1, 2, or 3" # for now, but could be extended to more power laws and even changed as emcee parameter!
        assert type(self.period_fine_grid) == np.ndarray, "Period grid requires a numpy array!"

    def __call__(self,low_period,high_period):
        return self.Period_pdf_area(low_period,high_period)

    def Period_pdf(self,Period):
        
        def power_law_1(P):
            return P**self.β1
        def power_law_2(P):
            return self.Period_break_1**(self.β1-self.β2)*P**self.β2
        def power_law_3(P):
            return self.Period_break_1**(self.β1-self.β2)*self.Period_break_2**(self.β2-self.β3)*P**self.β3
            
        if self.power_laws == 1:
            piecewise_func_list = [lambda P : power_law_1(P)]
            piecewise_conditions = [Period]
        elif self.power_laws == 2:
            piecewise_func_list = [lambda P : power_law_1(P),
                                   lambda P : power_law_2(P)
                                   ]
            piecewise_conditions = [Period<=self.Period_break_1, Period>self.Period_break_1]
        elif self.power_laws == 3:
            piecewise_func_list = [lambda P : power_law_1(P),
                                   lambda P : power_law_2(P),
                                   lambda P : power_law_3(P)
                                   ]
            piecewise_conditions = [Period<=self.Period_break_1, 
                                    (Period>self.Period_break_1) & (Period<=self.Period_break_2), 
                                    Period > self.Period_break_2]
        else: 
            print("this value of Period breaks is not supported")

        P_pdf = np.piecewise(Period,piecewise_conditions,piecewise_func_list)
        
        return P_pdf / np.trapezoid(P_pdf,Period) #Period)

    def Period_pdf_area(self,Period_lower, Period_upper):
        mask = (self.period_fine_grid > Period_lower) & (self.period_fine_grid <= Period_upper)
        return np.trapezoid(self.Period_pdf(self.period_fine_grid)[mask],self.period_fine_grid[mask])


class MassDistribution:
    def __init__(self,mass_fine_grid,mu_M,sigma_M):
        self.mass_fine_grid = mass_fine_grid
        self.mu_M = mu_M # be careful! this is in ln(M_E), not in M_E!
        self.sigma_M = sigma_M # same here, this is in ln(M_E), not in M_E!
        assert type(self.mass_fine_grid) == np.ndarray, "Mass grid requires a numpy array!"

    def __call__(self,low_mass,high_mass):
        return self.mass_pdf_area(low_mass,high_mass)
    
    def mass_pdf(self):
        """
        Returns the probability density function of the mass distribution.
        Uses a log-normal distribution.
        """
        # m_pdf = gamma_dist.pdf(self.mass_fine_grid, a=np.exp(self.ln_a), scale=np.exp(self.ln_beta))
        m_pdf = lognorm.pdf(self.mass_fine_grid, s=self.sigma_M, scale=np.exp(self.mu_M))
        # input()
        return (m_pdf) / np.trapezoid(m_pdf,self.mass_fine_grid)
    
    def mass_pdf_area(self,low_mass,high_mass):
        """
        Returns the area under the mass probability density function between low_mass and high_mass.
        """
        mask = (self.mass_fine_grid > low_mass) & (self.mass_fine_grid <= high_mass)
        return np.trapezoid(self.mass_pdf()[mask], self.mass_fine_grid[mask])


class RadiusDistribution:
    def __init__(self,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C):
        self.γ0 = γ0
        self.γ1 = γ1
        self.γ2 = γ2
        self.mass_break_1 = mass_break_1
        self.mass_break_2 = mass_break_2
        self.σ0 = σ0
        self.σ1 = σ1
        self.σ2 = σ2
        self.C = C

    def __call__(self,low_radius,high_radius):
        return self.radius_pdf_area(low_radius,high_radius)

    def _pure_silicate_radius(self,M):
        M1 = 10.55
        return 3.9 * 10**(-0.209594 + (1/3)*np.log10(M/M1) - 0.0799*(M/M1)**0.413)

    def _SN(self,M,mass_break_N):
        return 1 / (1 + np.exp(-5*(np.log(M)-np.log(mass_break_N))))

    def _mu0(self,M):
        return self.C*M**self.γ0

    def _mu1(self,M):
        return self.C*self.mass_break_1**(self.γ0-self.γ1)*M**self.γ1

    def _mu2(self,M):
        return self.C*self.mass_break_1**(self.γ0-self.γ1)*self.mass_break_2**(self.γ1-self.γ2)*M**self.γ2

    def mu_total(self,M,SN1,SN2):
        return ((1-SN1)*self._mu0(M) 
                + SN1*(1-SN2)*self._mu1(M)
                + SN1*SN2*self._mu2(M)
                )

    def sigma_total(self,SN1,SN2):
        return ((1-SN1)*self.σ0 
                + SN1*(1-SN2)*self.σ1
                + SN1*SN2*self.σ2
                )

    def sample_radius_given_mass(self,mass_distribution,rng):
        # recently rewritten...this version is 2x faster or so than the old truncnorm call. but see RadiusDistribution.ipynb for verification
        radii = np.empty_like(mass_distribution)

        SN1 = self._SN(mass_distribution,self.mass_break_1)
        SN2 = self._SN(mass_distribution,self.mass_break_2)
        
        mu = self.mu_total(mass_distribution,SN1,SN2)
        sigma = mu * self.sigma_total(SN1,SN2)
        
        lower_radius_bound = radius_given_density_mass(10, mass_distribution)
        a = (lower_radius_bound - mu) / sigma
        
        # Inverse CDF sampling — no rejection, O(N)
        Phi_a = ndtr(a)  # CDF at lower bound
        u = rng.uniform(0, 1, size=len(mass_distribution))
        radii = mu + sigma * ndtri(Phi_a + u * (1 - Phi_a))
        
        # print("min(radii),max(radii): ",min(radii),max(radii))
        # print("radii: ",radii)
        if not np.all(radii > 0.25):
            bad_places = np.where(radii <= 0.25)
            print("radii: ", radii)
            print("mu[bad_places]: ", mu[bad_places])
            print("sigma[bad_places]: ", sigma[bad_places])
            print("lower_density_bound[bad_places]: ", lower_radius_bound[bad_places])
            # print("radiis[np.where(radii <= 0.25)]: ", radii[np.where(radii <= 0.25)])
            raise ValueError("Radii must be above 0.25, but got radii = {}".format(radii))

        return radii
    
    def radius_pdf_area(self,low_radius,high_radius): # so does this just return the number of points in a certain radius range?
        """
        Returns the area under the radius probability density function between low_radius and high_radius.
        """
        radii = self.radius_pdf(self.mass_distribution)
        mask = (radii > low_radius) & (radii <= high_radius)
        return len(radii[mask]) / len(radii)
    

class EccentricityDistribution:
    def __init__(self, eccentricity_fine_grid, α=0.5, λ=1.0, σ=0.1):
        self.eccentricity_fine_grid = eccentricity_fine_grid
        assert type(self.eccentricity_fine_grid) == np.ndarray, "Eccentricity grid requires a numpy array!"
        self.α = α
        self.λ = λ
        self.σ = σ

    def __call__(self,low_e,high_e):
        return self.eccentricity_pdf_area(low_e,high_e)
    
    def rayleigh_exponential(self,e):
        return (self.α*((self.λ*np.exp(-self.λ*e))/(1-np.exp(-self.λ)))
              + (1-self.α)*((2*e*(1/(2*self.σ**2))*np.exp(-1*e**2/(2*self.σ**2)))/(1-np.exp(-1/(2*self.σ**2)))))

    def eccentricity_pdf(self, e): # note that this will probably get added on, with multiple rayleighs or gammas or betas
        """
        Returns the probability density function of the eccentricity distribution.
        """
        return self.rayleigh_exponential(e)
    
    def eccentricity_pdf_area(self, low_e, high_e):
        """
        Returns the area under the eccentricity probability density function between low_e and high_e.
        """
        mask = (self.eccentricity_fine_grid > low_e) & (self.eccentricity_fine_grid <= high_e)
        return np.trapezoid(self.eccentricity_pdf(self.eccentricity_fine_grid)[mask], self.eccentricity_fine_grid[mask])


def get_MES(stellar_df, mass, radius, period, ecc, omega, b):
    

    stellar_df["u1"] = -1.93 * 10**-4 * stellar_df['Teff'].iloc[0] + 1.5169
    stellar_df["u2"] = 1.25 * 10**-4 * stellar_df['Teff'].iloc[0] - 0.4601

    stellar_df["c0"] = 1 - (stellar_df['u1'].iloc[0] + stellar_df['u2'].iloc[0])
    stellar_df["omega_zink"] = stellar_df['c0'].iloc[0]/4 + (stellar_df['u1'].iloc[0]+(2*stellar_df['u2'].iloc[0]))/6 - stellar_df['u2'].iloc[0]/8

    # print("stellar median radius: ", np.median(stellar_df['radius']))
    # print(stellar_df)
    sm_axis = (G * (period*24*3600)**2 * (stellar_df['Mass'].iloc[0]*MSKG + mass*MEKG) / (4 * np.pi**2))**(1/3)  # semi-major axis in meters
    
    i = np.arccos(((1+ecc*np.sin(omega*np.pi/180))/(1-ecc**2))*(RSCM/100*stellar_df['Rad'].iloc[0]*b/sm_axis)) # check conversions here!
    
    k_rp = (RETORS*radius) / stellar_df['Rad'].iloc[0]
    
    n_tr = stellar_df["dataspan"].iloc[0] / period

    # print("period: ",period)
    # print("stellar_df['dataspan'].iloc[0]: ",stellar_df["dataspan"].iloc[0])
    # print("n_tr: ",n_tr)

    def get_transit_duration(period,b,ecc,i,omega,k_rp,sm_axis):
        # print("np.mean(stellar_df['radius']): ",np.median(stellar_df['radius']))
        # print("sm_axis: ",sm_axis)
        # print("np.sin(i): ",np.sin(i)) 
        # print(((RSCM/100)*np.median(stellar_df['radius'])/sm_axis) * np.sqrt((1+k_rp)**2 - b**2) / np.sin(i))
        # if abs((((RSCM/100)*stellar_df['radius'].iloc[0]/sm_axis) * np.sqrt((1+k_rp)**2 - b**2) / np.sin(i))) > 1:
        #     print("((RSCM/100)*stellar_df['radius']/sm_axis)", ((RSCM/100)*stellar_df['radius'].iloc[0]/sm_axis))
        #     print("stellar radius: ",stellar_df['radius'].iloc[0])
        #     print("k_rp: ",k_rp)
        #     print("b: ",b)
        #     print("sm_axis: ",sm_axis)
        #     print("np.sqrt((1+k_rp)**2 - b**2)", np.sqrt((1+k_rp)**2 - b**2))
        #     print("np.sin(i)", np.sin(i))
        #     print("np.sqrt(1-ecc**2) / (1+ecc*np.sin(omega*np.pi/180))", np.sqrt(1-ecc**2) / (1+ecc*np.sin(omega*np.pi/180)))
            
        #     raise ValueError("The abs argument of arcsin is greater than 1, which is not possible. Check your inputs.")
        arcsin_arg = np.clip(((RSCM/100)*stellar_df['Rad'].iloc[0]/sm_axis) * np.sqrt((1+k_rp)**2 - b**2) / np.sin(i) , -1, 1)
        return (period/np.pi) * np.arcsin(arcsin_arg) * np.sqrt(1-ecc**2) / (1+ecc*np.sin(omega*np.pi/180)) # check conversions here!

    def find_CDPP(transit_duration):
        
        
        cpdds = [stellar_df[col].iloc[0] for col in stellar_df.columns if col.startswith('rrmscdpp')]
        ###### TODO: doublecheck that the cpdds are in the right order with the durations
        durations = [1.5,2,2.5,3,3.5,4.5,5,6,7.5,9,10.5,12,12.5,15]
        cdpp_f = PchipInterpolator(durations,cpdds,extrapolate=False)

        # print("durations:", durations)
        # print("cpdds raw:", cpdds)
        # print("cpdds dtype/finite:", getattr(cpdds, 'dtype', None), np.isfinite(cpdds).all())
        # print("valid counts:", np.sum(np.isfinite(cpdds)))

        def cdpp_model(t, A, B):
            return np.sqrt((A**2) / t + B**2) # power law to extrapolate beyond the given transit duration regime
        
        # Fit this to your duration and CDPP values
        params, _ = curve_fit(cdpp_model, durations, cpdds)
        A, B = params
        return cdpp_f(transit_duration) if not np.isnan(cdpp_f(transit_duration)) else cdpp_model(transit_duration, A, B)
    
    def get_depth(stellar_df,k_rp):
        return 1 - (stellar_df['c0'].iloc[0]/4 
                    + ((stellar_df["u1"].iloc[0]+(2*stellar_df["u2"].iloc[0]))*(1-k_rp**2)**1.5)/6 
                    -   stellar_df["u2"].iloc[0]*(1-k_rp**2)/8) / (stellar_df["omega_zink"].iloc[0])


    
    # print("depth: ",get_depth(stellar_df,k_rp)*10**6)
    assert get_depth(stellar_df,k_rp)*10**6 > 0, "Depth must be greater than 0"
    
    if sm_axis < (RSCM/100)*stellar_df['Rad'].iloc[0]: #f"Semi-major axis {sm_axis} must be greater than stellar radius {(RSCM/100)*stellar_df["radius"].iloc[0]}"
        return np.nan, np.nan
    # print("i: ",i)

    # print("transit duration: ",get_transit_duration(period,b,ecc,i,omega,k_rp,sm_axis))
    # print("transit duration x 24 : ",get_transit_duration(period,b,ecc,i,omega,k_rp,sm_axis)*24)

    # print("n_tr: ",n_tr)
    # print("int(np.round(n_tr)): ", int(np.round(n_tr)))
    # # input()

    # print("c0: ",np.median(stellar_df["c0"]))

    # print("omega: ",np.median(stellar_df["omega"]))

    # print("CDPP: ",find_CDPP(get_transit_duration(period,b,ecc,i,omega,k_rp,sm_axis)*24))
    

    return (get_depth(stellar_df,k_rp)*10**6 / (find_CDPP(get_transit_duration(period,b,ecc,i,omega,k_rp,sm_axis)*24))) * 1.003 * n_tr**0.5, int(np.round(n_tr))
               

def get_transit_probability(stellar_df, mass, radius, period, ecc, omega):
    # geometric probability
    a = (G * (period*24*3600)**2 * (stellar_df["Mass"].iloc[0]*MSKG + mass*MEKG) / (4 * np.pi**2))**(1/3)  # semi-major axis in meters
    return np.min([1.0,((stellar_df["Rad"].iloc[0]*RSCM/100 + radius*RECM/100) / a) * ((1+ecc*np.sin(omega*np.pi/180))/(1-ecc**2))])


def get_detection_probability(MES,a=29.14,b=0.284,c=0.891):
    def integrand(x):
        return (c / (b**a * gamma(a)) ) * x**(a-1) * np.exp(-x/b)
    return quad(integrand, 0, MES)

# see hsu et al 2019
def get_detection_probability_hsu(MES,n_transits):
    match n_transits:
        case 3: a,b,c = 33.3884,0.264472,0.699093 
        case 4: a,b,c = 32.8860,0.269577,0.768366
        case 5: a,b,c = 31.5196,0.282741,0.833673
        case 6: a,b,c =	30.9919,0.286979,0.859865
        case _ if 7 <= n_transits <= 9: a,b,c = 30.1906,0.294688,0.875042
        case _ if 10 <= n_transits <= 18: a,b,c = 31.6342,0.279425,0.886144
        case _ if 19 <= n_transits <= 36: a,b,c = 32.6448,0.268898,0.889724
        case _ if 37 <= n_transits: a,b,c = 27.8185,0.32432,0.945075
        case _: 
            print("n_transits= ",n_transits)
            raise ValueError("n_transits is messed up...")
    
    def integrand(x):
        return (c / (b**a * gamma(a)) ) * x**(a-1) * np.exp(-x/b)
    return quad(integrand, 0, MES)

# def draw_radii(mass_distribution):

# generate a seed used to rng the synthetic catalog
def random_seed_generation(master_seed,*args):
    seed_seq = np.random.SeedSequence([int(master_seed)] + [int(a) for a in args])
    return int(seed_seq.generate_state(1)[0] & 0xFFFFFFFF)


def generate_catalog(stellar_info,p_Period, Period_fine_grid, p_mass, mass_fine_grid, γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid,rank,master_seed=None,time_seed=10):
    
    # np.random.seed(22)


    # print("begin generating fake catalog...")
    # print("len_stellar_df: ",len(stellar_df))

    begin_time = time.time()

    fake_catalog = np.zeros(((len_stellar_info:=len(stellar_info)),5),dtype=np.float32) # times 10 to test effects of undersampling
    # print("area under period distribution: ", np.trapezoid(p_Period, Period_fine_grid))
    # print("np.sum(p_Period): ", np.sum(p_Period))

    # print("fake_catalog shape: ", fake_catalog.shape)
    
    if master_seed is None:
        master_seed = 22
    # if time_seed is None:
    #     time_seed = int(time.time()) & 0xFFFFFF

    rng_metadata = {"master_seed":master_seed,
                    "rank_seed":rank,
                    "time_seed":time_seed} 

    rng_seed = random_seed_generation(master_seed,rank,time_seed)
    # print("rng_seed: ", rng_seed)
    rng = np.random.default_rng(seed=rng_seed)

    print("generation init time: ", (init_time:=time.time()) - begin_time)


    fake_catalog[:,0] = rng.choice(Period_fine_grid,size=len_stellar_info,p=p_Period)  # Period

    print("period gen time: ", (period_gen_time:=time.time()) - init_time)


    fake_catalog[:,1] = rng.choice(mass_fine_grid,size=len_stellar_info,p=p_mass)  # Mass
    mask = fake_catalog[:,1] < 0.1
    while np.any(mask):
        print("Some masses are less than 0.1 M_E, regenerating...")
        fake_catalog[:,1][mask] = rng.choice(mass_fine_grid,size=len(fake_catalog[:,1][mask]),p=p_mass)

    # print("number of mass 4 - 24: ", np.sum((fake_catalog[:,1] > 4) & (fake_catalog[:,1] < 24)))
    
    print("mass gen time: ", (mass_gen_time:=time.time()) - period_gen_time)

    
    # print("number of M greater than 5000: ", np.sum(fake_catalog[:,1]>5000))    
    
    # print("make radius distribution...")
    fake_catalog[:,2] = RadiusDistribution(γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C).sample_radius_given_mass(fake_catalog[:,1],rng)  # Radius
    # fake_catalog[:,2] = np.random.choice(fake_catalog[:,1],size=len_stellar_info,p=p_radius)  # Radius THIS NEEDS EDITING RADIUS IS WEIRD
    print("radius gen time: ", (radius_gen_time:=time.time()) - mass_gen_time)
    
    fake_catalog[:,3] = rng.choice(eccentricity_fine_grid,size=len_stellar_info,p=p_ecc)  # Eccentricity

    print("ecc gen time: ", (ecc_gen_time:=time.time()) - radius_gen_time)

    fake_catalog[:,4] = rng.uniform(0,360,len_stellar_info)  # omega (argument of periastron)
    # fake_catalog[:,5] = np.random.uniform(-1,1,len_stellar_df)  # b (impact parameter) ... do we need this? why do we need it?
    
    print("omega gen time: ", (omega_gen_time:=time.time()) - ecc_gen_time)

    return fake_catalog, rng_metadata


def get_probability_distributions(params):
    # unpack params
    γ0 = params[1]
    γ1 = params[2]
    γ2 = params[3]
    σ0 = params[4]
    σ1 = params[5]
    σ2 = params[6]
    mass_break_1 = params[7]
    mass_break_2 = params[8]
    C = params[9]
    mu_M = params[10]
    sigma_M = params[11]
    β1 = params[12]
    β2 = params[13]
    # β3 = params[14]
    Period_break_1 = params[14]
    # Period_break_2 = params[16]
    α = params[15]
    λ = params[16]
    σ_e = params[17]

    # period
    Period_fine_grid = np.linspace(0.1,500,10000,dtype=np.float32)
    pdf_Period = PeriodDistribution(Period_fine_grid,[β1,β2],[Period_break_1],power_laws=2).Period_pdf(Period_fine_grid)
    pmf_Period = normalize_pdf_to_pmf(pdf_Period,Period_fine_grid)
    # p_Period = normalize_pdf_to_pmf(pdf_Period, Period_fine_grid)

    # mass
    mass_fine_grid = np.logspace(-1,4,10000,dtype=np.float32) # used to be np.linspace(.1,10000,100000) that might be right?
    pdf_mass = MassDistribution(mass_fine_grid,mu_M,sigma_M).mass_pdf()
    pmf_mass = normalize_pdf_to_pmf(pdf_mass,mass_fine_grid)

    # print("pmass: ", p_mass)
    # print("area under mass distribution: ", np.trapezoid(pdf_mass, mass_fine_grid))
    
    # radius 

    # ecc
    eccentricity_grid = np.linspace(0,1,10000,dtype=np.float32)
    pdf_ecc = EccentricityDistribution(eccentricity_grid,α,λ,σ_e).eccentricity_pdf(eccentricity_grid)
    pmf_ecc = normalize_pdf_to_pmf(pdf_ecc, eccentricity_grid)
    # print("p_ecc: ", p_ecc)
    # print("alpha: ", α)
    # print("lambda: ", λ)
    # print("sigma_e: ", σ_e)
    # print("area under eccentricity distribution: ", np.trapezoid(p_ecc, eccentricity_grid))    

    
    if (is_nan_in_pmfs := (np.isnan(pmf_ecc).any() or np.isnan(pmf_Period).any() or np.isnan(pmf_mass).any())):
        print("Warning: PMFs contain NaN. This parameter draw is bad, let's skip it!")

    if (is_inf_in_pmfs := (not np.isfinite(pmf_ecc).any() or not np.isfinite(pmf_Period).any() or not np.isfinite(pmf_mass).any())):
        print("Warning: PMFs contain inf. This parameter draw is bad, let's skip it!")

    if (is_neg_in_pmfs := (np.any(pmf_ecc < 0) or np.any(pmf_Period < 0) or np.any(pmf_mass < 0))):
        print("Warning: PMFs contain negative values. This parameter draw is bad, let's skip it!")



    return pmf_Period, Period_fine_grid, pmf_mass, mass_fine_grid,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, pmf_ecc, eccentricity_grid, is_nan_in_pmfs, is_inf_in_pmfs, is_neg_in_pmfs


def normalize_pdf_to_pmf(pdf, grid):
    dx = np.gradient(grid)
    pmf = pdf * dx
    pmf /= np.sum(pmf)
    return pmf


def synthetic_catalog_to_grid(synthetic_catalog, voxel_grid, synthetic_multiplier):
    print("synthetic catalog original shape: ", synthetic_catalog.shape)
    # print("synthetic catalog head: ", synthetic_catalog[:5,:])
    # print("synthetic catalog count: ", np.sum(synthetic_catalog))
    # originally synthetic catalog is in order period, mass, radius, ecc, omega confirmed 12/19 that this is working right
    synthetic_catalog = synthetic_catalog[:, [2, 0, 1, 3, 4]]

    # print("rearranged catalog: ", synthetic_catalog)
    synthetic_catalog = synthetic_catalog[
        ~((synthetic_catalog[:, 0] < np.min(voxel_grid.radius_grid_array)) |
        (synthetic_catalog[:, 0] > np.max(voxel_grid.radius_grid_array)))
        ]
    synthetic_catalog = synthetic_catalog[
        ~((synthetic_catalog[:,1] < np.min(voxel_grid.period_grid_array)) |
        (synthetic_catalog[:,1] > np.max(voxel_grid.period_grid_array)))
        ]
    synthetic_catalog = synthetic_catalog[
        ~((synthetic_catalog[:,2] < np.min(voxel_grid.mass_grid_array)) |
        (synthetic_catalog[:,2] > np.max(voxel_grid.mass_grid_array)))
        ]  
    synthetic_catalog = synthetic_catalog[
        ~((synthetic_catalog[:,3] < np.min(voxel_grid.eccentricity_grid_array)) |
        (synthetic_catalog[:,3] > np.max(voxel_grid.eccentricity_grid_array)))
        ]      
    synthetic_catalog = synthetic_catalog[
        ~((synthetic_catalog[:,4] < np.min(voxel_grid.omega_grid_array)) |
        (synthetic_catalog[:,4] > np.max(voxel_grid.omega_grid_array)))
        ]    
    
    ### need to implement a "realistic" filter
    ### wherein planets that go too close to their star (and possibly do anything else unphysical) are removed
    



    ### RegGridInterp is 5x or worse slower than map_interp, and gives identical results, we're removing it
    # before_reg_interp_time = time.time()
    # completeness = voxel_grid.completeness_interp(synthetic_catalog)
    # print("RegGridInterp time: ", (after_reg_interp_time:=time.time()) - before_reg_interp_time)

    before_map_interp_time = time.time()
    completeness = voxel_grid.interpolate_completeness(synthetic_catalog)
    print("map interp time: ", (after_map_interp_time:=time.time()) - before_map_interp_time)


    # print("max(abs(diff)) for map vs RegGridInterp methods :", np.max(np.abs(completeness - completeness_alt)))
    
    ## the max abs diff for the two is 10e-15, seems like these methods are essentially the same




    # print("completeness shape: ", completeness.shape)
    # print("completeness head: ", completeness[:5])
    # print("completeness min/max/mean:", completeness.min(), completeness.max(), completeness.mean())
    # print("completeness near-zero fraction:", np.mean(completeness < 0.1))
    # print("completeness above-one fraction:", np.mean(completeness > 1))
    # print("completeness nan count:", np.sum(np.isnan(completeness)))

    bins = [
        radius_grid_array,
        period_grid_array,
        mass_grid_array,
        eccentricity_grid_array,
        omega_grid_array
    ]

    histtestsums, edges = np.histogramdd(synthetic_catalog, bins=bins, weights=completeness)
    histtestsums /= synthetic_multiplier

    voxel_grid.likelihood_array[:,:,:,:,:, 1] = histtestsums
    # packpointssums =  pack_points_vectorized(synthetic_catalog,voxel_grid,completeness).likelihood_array[:,:,:,:,:,1]
    
    # assert np.testing.assert_array_almost_equal(packpointssums, histtestsums), "similarity test FAILED"
    ## if this is true, we should get rid of pack_points vectorized sheesh
    ## after tests, only 50/2e6 voxels failed:
    # Mismatched elements: 50 / 2010960 (0.00249%)
    # Max absolute difference among violations: 0.00012738
    # Max relative difference among violations: 0.08264412
    # a tiny error that results from edge handling, we are just going with histogramdd.

    return voxel_grid