import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.stats import lognorm, truncnorm
from scipy.special import gamma

from kg_constants import G, RETORS, RSCM, MSKG, MEKG, RECM, RSCM
from kg_utilities import radius_given_density_mass


class PeriodDistribution:
    def __init__(self, period_fine_grid, β1, β2, β3, Period_break_1, Period_break_2, power_laws=3):
        self.period_fine_grid = period_fine_grid
        self.β1 = β1
        self.β2 = β2
        self.β3 = β3
        self.Period_break_1 = Period_break_1
        self.Period_break_2 = Period_break_2
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
        
        return P_pdf / np.trapezoid(P_pdf,Period)

    def Period_pdf_area(self,Period_lower, Period_upper):
        mask = (self.period_fine_grid > Period_lower) & (self.period_fine_grid <= Period_upper)
        return np.trapezoid(self.Period_pdf(self.period_fine_grid)[mask],self.period_fine_grid[mask])


class MassDistribution:
    def __init__(self,mass_fine_grid,μ,σ):
        self.mass_fine_grid = mass_fine_grid
        self.μ = μ # be careful! this is in ln(M_E), not in M_E!
        self.σ = σ # same here, this is in ln(M_E), not in M_E!
        assert type(self.mass_fine_grid) == np.ndarray, "Mass grid requires a numpy array!"

    def __call__(self,low_mass,high_mass):
        return self.mass_pdf_area(low_mass,high_mass)
    
    def mass_pdf(self):
        """
        Returns the probability density function of the mass distribution.
        Uses a log-normal distribution.
        """
        m_pdf = lognorm.pdf(self.mass_fine_grid, s=self.σ, scale=np.exp(self.μ))
        # input()
        return (m_pdf) / np.trapezoid(m_pdf,self.mass_fine_grid)
    
    def mass_pdf_area(self,low_mass,high_mass):
        """
        Returns the area under the mass probability density function between low_mass and high_mass.
        """
        mask = (self.mass_fine_grid > low_mass) & (self.mass_fine_grid <= high_mass)
        return np.trapezoid(self.mass_pdf()[mask], self.mass_fine_grid[mask])


class RadiusDistribution:
    def __init__(self,mass_fine_grid,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C):
        self.mass_fine_grid = mass_fine_grid # one question: what exactly...is this?
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

    def mu_total(self,M):
        return ((1-self._SN(M,self.mass_break_1))*self._mu0(M) 
                + self._SN(M,self.mass_break_1)*(1-self._SN(M,self.mass_break_2))*self._mu1(M)
                + self._SN(M,self.mass_break_1)*self._SN(M,self.mass_break_2)*self._mu2(M)
                )

    def sigma_total(self,M):
        return ((1-self._SN(M,self.mass_break_1))*self.σ0 
                + self._SN(M,self.mass_break_1)*(1-self._SN(M,self.mass_break_2))*self.σ1
                + self._SN(M,self.mass_break_1)*self._SN(M,self.mass_break_2)*self.σ2
                )

    def radius_pdf(self,masses):
        radii = np.empty_like(masses)

        mu = self.mu_total(masses)
        sigma = mu * self.sigma_total(masses)
        # print("sigma: ", sigma)
        # print("mu: ", mu)
        if not np.all(sigma > 0):
            print("self.γ0",self.γ0)
            print("self.γ1",self.γ1)
            print("self.γ2",self.γ2)
            print("self.mass_break_1",self.mass_break_1)
            print("self.mass_break_2",self.mass_break_2)
            print("self.σ0",self.σ0)
            print("self.σ1",self.σ1)
            print("self.σ2",self.σ2)
            print("self.C",self.C) 
        assert np.all(sigma > 0), "Sigma must be positive, but got sigma = {}".format(sigma)
        lower_density_bound = radius_given_density_mass(10, masses) # this is the upper density limit of 10 g/cm^3
        a = (lower_density_bound - mu) / sigma
        b = np.full_like(a, np.inf)
        radii = truncnorm.rvs(a,b, loc=mu, scale=sigma)
        if not np.all(radii > 0.25):
            bad_places = np.where(radii <= 0.25)
            print("radii: ", radii)
            print("mu[bad_places]: ", mu[bad_places])
            print("sigma[bad_places]: ", sigma[bad_places])
            print("lower_density_bound[bad_places]: ", lower_density_bound[bad_places])
            # print("radiis[np.where(radii <= 0.25)]: ", radii[np.where(radii <= 0.25)])
            raise ValueError("Radii must be above 0.25, but got radii = {}".format(radii))
        # assert np.all(radii > 0.25), "Radii must be above 0.25, but got radii = {}".format(radii)
        # radii = np.random.normal(mu, sigma)
        # mask = radii < 0.25
        # i = 0
        # while np.any(mask):
        #     radii[mask] = np.random.normal(mu[mask], sigma[mask])
        #     mask = radii < 0.25
        #     if i%10 ==0:
        #         print("mu[mask]: ", mu[mask])
        #         print("sigma[mask]: ", sigma[mask])
        #         print("radii[mask]: ", radii[mask])
        #         print(i," iterations to get radii > 0.25")
        #         print("len(radii[mask]): ", len(radii[mask]))
        #     if i%100 == 0:
        #         print("len of mass < 0.1: ", len(masses[masses < 0.1]))
        #         print(masses)
        #         print("radii : ", radii)
        #     i += 1
            
        # for i,mass in enumerate(masses):
        #     # if (radius := pure_silicate_radius(mass)) < 1.6 and mass < 100:
        #     #     radii = np.append(radii,radius)
        #     # else:
        #     radius = np.random.normal(mu_tot:=self.mu_total(mass),mu_tot*self.sigma_total(mass))
        #     while radius < 0.4:
        #         radius = np.random.normal(mu_tot,mu_tot*self.sigma_total(mass))
        #     radii[i] = radius
        return radii
    
    def radius_pdf_area(self,low_radius,high_radius): # so does this just return the number of points in a certain radius range?
        """
        Returns the area under the radius probability density function between low_radius and high_radius.
        """
        radii = self.radius_pdf(self.mass_fine_grid)
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
    
    # stellar_df["u1"] = stellar_df['limbdark_coeff1'] #-1.93 * 10**-4 * stellar_df['teff'] + 1.5169
    # stellar_df["u2"] = stellar_df['limbdark_coeff2'] #1.25 * 10**-4 * stellar_df['teff'] - 0.4601



    stellar_df["u1"] = -1.93 * 10**-4 * stellar_df['teff'] + 1.5169
    stellar_df["u2"] = 1.25 * 10**-4 * stellar_df['teff'] - 0.4601

    stellar_df["c0"] = 1 - (stellar_df['u1'] + stellar_df['u2'])
    stellar_df["omega"] = stellar_df['c0']/4 + (stellar_df['u1']+(2*stellar_df['u2']))/6 - stellar_df['u2']/8

    # print("stellar median radius: ", np.median(stellar_df['radius']))
    # print(stellar_df)
    sm_axis = (G * (period*24*3600)**2 * (np.median(stellar_df['mass'])*MSKG + mass*MEKG) / (4 * np.pi**2))**(1/3)  # semi-major axis in meters
    
    i = np.arccos(((1+ecc*np.sin(omega*np.pi/180))/(1-ecc**2))*(RSCM/100*np.median(stellar_df['radius'])*b/sm_axis)) # check conversions here!
    
    k_rp = (RETORS*radius) / np.median(stellar_df['radius'])
    
    n_tr = np.median(stellar_df["dataspan"]) / period


    def get_transit_duration(period,b,ecc,i,omega,k_rp,sm_axis):
        # print("np.mean(stellar_df['radius']): ",np.median(stellar_df['radius']))
        # print("sm_axis: ",sm_axis)
        # print("np.sin(i): ",np.sin(i)) 
        # print(((RSCM/100)*np.median(stellar_df['radius'])/sm_axis) * np.sqrt((1+k_rp)**2 - b**2) / np.sin(i))
        if abs((((RSCM/100)*np.median(stellar_df['radius'])/sm_axis) * np.sqrt((1+k_rp)**2 - b**2) / np.sin(i)) * np.sqrt(1-ecc**2) / (1+ecc*np.sin(omega*np.pi/180))) > 1:
            print("((RSCM/100)*np.median(stellar_df['radius'])/sm_axis)", ((RSCM/100)*np.median(stellar_df['radius'])/sm_axis))
            print("np.sqrt((1+k_rp)**2 - b**2)", np.sqrt((1+k_rp)**2 - b**2))
            print("np.sin(i)", np.sin(i))
            print("np.sqrt(1-ecc**2) / (1+ecc*np.sin(omega*np.pi/180))", np.sqrt(1-ecc**2) / (1+ecc*np.sin(omega*np.pi/180)))
            
            raise ValueError("The abs argument of arcsin is greater than 1, which is not possible. Check your inputs.")
        
        return (period/np.pi) * np.arcsin(((RSCM/100)*np.median(stellar_df['radius'])/sm_axis) * np.sqrt((1+k_rp)**2 - b**2) / np.sin(i)) * np.sqrt(1-ecc**2) / (1+ecc*np.sin(omega*np.pi/180)) # check conversions here!

    def find_CDPP(transit_duration):
        
        # cpdds = [np.median(stellar_df[col]) for col in stellar_df.columns if col.startswith('rrmscdpp')]
        # durations = [1.5,2,2.5,3,3.5,4.5,5,6,7.5,9,10.5,12,12.5,15]
        # cdpp_f = UnivariateSpline(durations,cpdds,s=0)
        # return cdpp_f(transit_duration)
            # def find_CDPP(transit_duration):
        
        cpdds = [np.median(stellar_df[col]) for col in stellar_df.columns if col.startswith('rrmscdpp')]
        durations = [1.5,2,2.5,3,3.5,4.5,5,6,7.5,9,10.5,12,12.5,15]
        # cdpp_f = UnivariateSpline(durations,cpdds,s=0)
        cdpp_f = PchipInterpolator(durations,cpdds,extrapolate=False)


        def cdpp_model(t, A, B):
            return np.sqrt((A**2) / t + B**2) # power law to extrapolate beyond the given transit duration regime
        
        # Fit this to your duration and CDPP values
        params, _ = curve_fit(cdpp_model, durations, cpdds)
        A, B = params
        return cdpp_f(transit_duration) if not np.isnan(cdpp_f(transit_duration)) else cdpp_model(transit_duration, A, B)
    
    def get_depth(stellar_df,k_rp):
        return 1 - (np.median(stellar_df['c0'])/4 
                    + ((np.median(stellar_df["u1"])+(2*np.median(stellar_df["u2"])))*(1-k_rp**2)**1.5)/6 
                    -   np.median(stellar_df["u2"])*(1-k_rp**2)/8) / (np.median(stellar_df["omega"]))


    
    # print("depth: ",get_depth(stellar_df,k_rp)*10**6)
    assert get_depth(stellar_df,k_rp)*10**6 > 0, "Depth must be greater than 0"
    assert sm_axis > (RSCM/100)*np.median(stellar_df['radius']), "Semi-major axis must be greater than stellar radius"
    
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
    a = (G * (period*24*3600)**2 * (np.median(stellar_df["mass"])*MSKG + mass*MEKG) / (4 * np.pi**2))**(1/3)  # semi-major axis in meters
    return ((np.median(stellar_df["radius"])*RSCM/100 + radius*RECM/100) / a) * ((1+ecc*np.sin(omega*np.pi/180))/(1-ecc**2))


def get_detection_probability(MES,a=29.14,b=0.284,c=0.891):
    def integrand(x):
        return (c / (b**a * gamma(a)) ) * x**(a-1) * np.exp(-x/b)
    return quad(integrand, 0, MES)

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
        case _: raise ValueError("n_transits is messed up...")
    
    def integrand(x):
        return (c / (b**a * gamma(a)) ) * x**(a-1) * np.exp(-x/b)
    return quad(integrand, 0, MES)

# def draw_radii(mass_distribution):



def generate_catalog(stellar_df, p_Period, Period_fine_grid, p_mass, mass_fine_grid, γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_fine_grid):
    fake_catalog = np.zeros(((len_stellar_df:=len(stellar_df)),6))
    # print("area under period distribution: ", np.trapezoid(p_Period, Period_fine_grid))
    # print("np.sum(p_Period): ", np.sum(p_Period))
    # print("begin generating fake catalog...")
    fake_catalog[:,0] = np.random.choice(Period_fine_grid,size=len_stellar_df,p=p_Period)  # Period

    fake_catalog[:,1] = np.random.choice(mass_fine_grid,size=len_stellar_df,p=p_mass)  # Mass
    mask = fake_catalog[:,1] < 0.1
    while np.any(mask):
        print("Some masses are less than 0.1 M_E, regenerating...")
        fake_catalog[:,1][mask] = np.random.choice(mass_fine_grid,size=len(fake_catalog[:,1][mask]),p=p_mass)
    
    # print("make radius distribution...")
    fake_catalog[:,2] = RadiusDistribution(fake_catalog[:,1],γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C).radius_pdf(fake_catalog[:,1])  # Radius
    # fake_catalog[:,2] = np.random.choice(fake_catalog[:,1],size=len_stellar_df,p=p_radius)  # Radius THIS NEEDS EDITING RADIUS IS WEIRD
    
    fake_catalog[:,3] = np.random.choice(eccentricity_fine_grid,size=len_stellar_df,p=p_ecc)  # Eccentricity
    fake_catalog[:,4] = np.random.uniform(0,2*np.pi,len_stellar_df)  # omega (argument of periastron)
    fake_catalog[:,5] = np.random.uniform(-1,1,len_stellar_df)  # b (impact parameter)
    # print("fake catalog has been created!")
    return fake_catalog


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
    μM = params[10]
    σM = params[11]
    β1 = params[12]
    β2 = params[13]
    β3 = params[14]
    Period_break_1 = params[15]
    Period_break_2 = params[16]
    α = params[17]
    λ = params[18]
    σ_e = params[19]

    # period
    Period_fine_grid = np.linspace(0.1,500,10000)
    pdf_Period = PeriodDistribution(Period_fine_grid,β1,β2,β3,Period_break_1,Period_break_2).Period_pdf(Period_fine_grid)
    p_Period = normalize_pdf_to_pmf(pdf_Period, Period_fine_grid)

    # mass
    mass_fine_grid = np.logspace(-1,4,10000)
    pdf_mass = MassDistribution(mass_fine_grid,μM,σM).mass_pdf()
    p_mass = normalize_pdf_to_pmf(pdf_mass, mass_fine_grid)
    # print("pmass: ", p_mass)
    # print("area under mass distribution: ", np.trapezoid(pdf_mass, mass_fine_grid))
    
    # radius 

    # ecc
    eccentricity_grid = np.linspace(0,1,10000)
    pdf_ecc = EccentricityDistribution(eccentricity_grid,α,λ,σ_e).eccentricity_pdf(eccentricity_grid)
    p_ecc = normalize_pdf_to_pmf(pdf_ecc, eccentricity_grid)
    # print("p_ecc: ", p_ecc)
    # print("alpha: ", α)
    # print("lambda: ", λ)
    # print("sigma_e: ", σ_e)
    # print("area under eccentricity distribution: ", np.trapezoid(p_ecc, eccentricity_grid))    

    return p_Period, Period_fine_grid, p_mass, mass_fine_grid,γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C, p_ecc, eccentricity_grid


def normalize_pdf_to_pmf(pdf, grid):
    """
    Converts a continuous PDF sampled on a grid to a PMF usable in np.random.choice.
    
    Parameters:
        pdf (np.ndarray): PDF values evaluated on the grid.
        grid (np.ndarray): Grid values corresponding to the PDF.
    
    Returns:
        np.ndarray: A PMF (probability weights) that sums to 1.
    """
    dx = np.diff(grid)
    # print("dx: ", dx)
    dx = np.append(dx, dx[-1])  # Extend last interval to preserve length
    pmf = pdf * dx
    pmf /= np.sum(pmf)
    # print("pmf: ", pmf)
    if np.isnan(pmf).any():
        print("Warning: PMF contains NaN values. This may indicate an issue with the PDF or grid.")
    return pmf



def voxel_model_count(voxel_grid,voxel,synthetic_catalog):
    
    mask = ((synthetic_catalog[:,0] >= voxel.bottom_period) & (synthetic_catalog[:,0] < voxel.top_period) 
            & (synthetic_catalog[:,1] >= voxel.bottom_mass) & (synthetic_catalog[:,1] < voxel.top_mass) 
            & (synthetic_catalog[:,2] >= voxel.bottom_radius) & (synthetic_catalog[:,2] < voxel.top_radius)
            & (synthetic_catalog[:,3] >= voxel.bottom_eccentricity) & (synthetic_catalog[:,3] < voxel.top_eccentricity)
            & (synthetic_catalog[:,4] >= voxel.bottom_omega) & (synthetic_catalog[:,4] < voxel.top_omega) 
            ) 
    
    catalog_in_voxel = synthetic_catalog[mask]

    points = np.column_stack([
        catalog_in_voxel[:,2],  # radius
        catalog_in_voxel[:,0],  # period
        catalog_in_voxel[:,1],  # mass
        catalog_in_voxel[:,3],  # ecc
        catalog_in_voxel[:,4]   # omega
    ])

    return np.sum(voxel_grid.p_detection_interp(points)   
                * voxel_grid.p_transit_interp(points)  
                  )