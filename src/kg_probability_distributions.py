import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.stats import lognorm



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

    def _solve_for_A(self,objective,Period,power_law_1,power_law_2, power_law_3):
        sol = root_scalar(objective, bracket=[1e-5,2],method='brentq',args=(Period,power_law_1,power_law_2,power_law_3,))
        if sol.converged:            #for brentq: bracket = [1e-6,2] instead of x0
            return sol.root
        else:
            print("unable to solve!")

    def _period_normalization_constant(self,A,period,power_law_1,power_law_2,power_law_3):
        return self._integral_of_period_pdf(A,period,power_law_1,power_law_2,power_law_3) - 1

    def _integral_of_period_pdf(self,A,period,power_law_1,power_law_2,power_law_3):
        lower_bound = np.min(period)
        upper_bound = np.max(period)
        
        power_law_1_upper_bound = self.Period_break_1 if self.power_laws > 1 else upper_bound
        power_law_2_upper_bound = self.Period_break_2 if self.power_laws > 2 else upper_bound

        error_tol = 1e-3
        limit = 100
        result1,_ = quad(power_law_1,lower_bound,power_law_1_upper_bound,args=(A),epsabs=error_tol,epsrel=error_tol,limit=limit)
        if self.power_laws == 1 : return result1
        result2,_ = quad(power_law_2,self.Period_break_1,power_law_2_upper_bound,args=(A),epsabs=error_tol,epsrel=error_tol,limit=limit)
        if self.power_laws == 2: return result1+result2
        result3,_ = quad(power_law_3,self.Period_break_2,upper_bound, args=(A),epsabs=error_tol,epsrel=error_tol,limit=limit)
        if self.power_laws == 3: return result1+result2+result3
        else:
            print("this number of power laws is not yet supported!")

    def Period_pdf(self,Period):
        
        def power_law_1(P,A):
            return A*P**self.β1
        def power_law_2(P,A):
            return A*self.Period_break_1**(self.β1-self.β2)*P**self.β2
        def power_law_3(P,A):
            return A*self.Period_break_1**(self.β1-self.β2)*self.Period_break_2**(self.β2-self.β3)*P**self.β3
            
        A = self._solve_for_A(self._period_normalization_constant,Period,power_law_1,power_law_2,power_law_3)
        if self.power_laws == 1:
            piecewise_func_list = [lambda P : power_law_1(P,A)]
            piecewise_conditions = [Period]
        elif self.power_laws == 2:
            piecewise_func_list = [lambda P : power_law_1(P,A),lambda P : power_law_2(P,A)]
            piecewise_conditions = [Period<=self.Period_break_1, Period>self.Period_break_1]
        elif self.power_laws == 3:
            piecewise_func_list = [lambda P : power_law_1(P,A),lambda P : power_law_2(P,A),lambda P : power_law_3(P,A)]
            piecewise_conditions = [Period<=self.Period_break_1, (Period>self.Period_break_1) & (Period<=self.Period_break_2), Period > self.Period_break_2]
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
        self.μ = μ
        self.σ = σ
        assert type(self.mass_fine_grid) == np.ndarray, "Mass grid requires a numpy array!"

    def __call__(self,low_mass,high_mass):
        return self.mass_pdf_area(low_mass,high_mass)
    
    def mass_pdf(self):
        """
        Returns the probability density function of the mass distribution.
        Uses a log-normal distribution.
        """
        return (m_pdf:=lognorm.pdf(self.mass_fine_grid, s=self.σ, scale=self.μ)) / np.trapezoid(m_pdf,self.mass_fine_grid)
    
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
        return ((1-self._SN(M,self.mass_break_1))*self._mu0(M) + 
                self._SN(M,self.mass_break_1)*(1-self._SN(M,self.mass_break_2))*self._mu1(M) +
                self._SN(M,self.mass_break_1)*self._SN(M,self.mass_break_2)*self._mu2(M)
                )

    def sigma_total(self,M):
        return ((1-self._SN(M,self.mass_break_1))*self.σ0 + 
                self._SN(M,self.mass_break_1)*(1-self._SN(M,self.mass_break_2))*self.σ1 +
                self._SN(M,self.mass_break_1)*self._SN(M,self.mass_break_2)*self.σ2
                )

    def radius_pdf(self,masses):
        radii = np.empty(0)
        for mass in masses:
            # if (radius := pure_silicate_radius(mass)) < 1.6 and mass < 100:
            #     radii = np.append(radii,radius)
            # else:
                radius = np.random.normal(self.mu_total(mass),self.sigma_total(mass))
                radii = np.append(radii,radius)
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
        return (self.α*((self.λ*np.exp(-self.λ*e))/(1-np.exp(-self.λ))) + 
            (1-self.α)*((2*e*(1/(2*self.σ**2))*np.exp(-1*e**2/(2*self.σ**2)))/(1-np.exp(-1/(2*self.σ**2)))))

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

def voxel_model_count(voxel,params):
    
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

    print("C: ",C)

    # period
    Period_grid = np.linspace(0.1,500,10000)
    p_Period = PeriodDistribution(Period_grid,β1,β2,β3,Period_break_1,Period_break_2)

    # mass
    mass_grid = np.logspace(-1,4,10000)
    p_mass = MassDistribution(mass_grid,μM,σM)

    # radius 
    p_radius = RadiusDistribution(p_mass.mass_pdf(),γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C)

    # ecc
    eccentricity_grid = np.linspace(0,1,10000)
    p_ecc = EccentricityDistribution(eccentricity_grid,α,λ,σ_e)

    return (p_Period(voxel.bottom_period,voxel.top_period)
            * p_mass(voxel.bottom_mass,voxel.top_mass)
            * p_radius(voxel.bottom_radius,voxel.top_radius)
            * p_ecc(voxel.bottom_eccentricity,voxel.top_eccentricity)
            )
    