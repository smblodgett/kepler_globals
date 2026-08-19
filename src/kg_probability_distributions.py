import numpy as np
import pandas as pd
from numba import njit
import time
import warnings

from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.stats import lognorm, norm # truncnorm #, gaussian_kde
# from scipy.stats import gamma as gamma_dist
from scipy.special import gamma, gammaln, gammainc, logsumexp, ndtr, ndtri


from kg_constants import G, RETORS, RSCM, MSKG, MEKG, RECM, RSCM
from kg_utilities import radius_given_density_mass, density_given_mass_radius
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

    def mu_total(self, M, S1=None, S2=None):
        S1 = S1 if S1 is not None else self._SN(M, self.mass_break_1)
        S2 = S2 if S2 is not None else self._SN(M, self.mass_break_2)
        return ((1-S1)*self._mu0(M) + S1*(1-S2)*self._mu1(M) + S1*S2*self._mu2(M))

    def sigma_total(self, M, S1=None, S2=None):
        S1 = S1 if S1 is not None else self._SN(M, self.mass_break_1)
        S2 = S2 if S2 is not None else self._SN(M, self.mass_break_2)
        return ((1-S1)*self.σ0 + S1*(1-S2)*self.σ1 + S1*S2*self.σ2)

    def sample_radius_given_mass(self,mass_distribution,rng):
        # recently rewritten...this version is 2x faster or so than the old truncnorm call. but see RadiusDistribution.ipynb for verification
        t = time.time()

        # mass_distribution = mass_distribution # needs to be float64 or ndtr() and ndtri() have overflow/underflow issues
        # print("cast time: ", (cast_time:=time.time()) - t)

        S1 = self._SN(mass_distribution, self.mass_break_1)
        S2 = self._SN(mass_distribution, self.mass_break_2)
        mu = self.mu_total(mass_distribution, S1, S2)
        sigma = mu * self.sigma_total(mass_distribution, S1, S2)
        # print("SN mu and sigma time: ", (SN_mu_simga_time:=time.time()) - cast_time)

        
        lower_radius_bound = radius_given_density_mass(10, mass_distribution)
        a = (lower_radius_bound - mu) / sigma
        del lower_radius_bound

        # print("lrb and a time: ", (lrb_a_time:=time.time()) - SN_mu_simga_time)

        
        # Inverse CDF sampling — no rejection, O(N)
        Phi_a = ndtr(a)  # CDF at lower bound

        ## tiny underflow slowing hypothesis
        # tiny = np.finfo(np.float64).tiny  # smallest normal double, ~2.2e-308
        # n_denormal = np.sum((Phi_a > 0) & (Phi_a < tiny))

        u = rng.uniform(0, 1, size=len(mass_distribution))
        radii = mu + sigma * ndtri(Phi_a + u * (1 - Phi_a))

        # print("radii calc time: ", (radii_time:=time.time()) - lrb_a_time) #,f"a range: [{a.min():.1f}, {a.max():.1f}]  denormal Phi_a count: {n_denormal}")

        # print("min(radii),max(radii): ",min(radii),max(radii))
        # print("radii: ",radii)
        if not np.all(radii > 0.25):
            bad_places = np.where(radii <= 0.25)
            print("radii: ", radii)
            print("mu[bad_places]: ", mu[bad_places])
            print("sigma[bad_places]: ", sigma[bad_places])
            print("lower_density_bound[bad_places]: ", lower_radius_bound[bad_places])
            # print("radiis[np.where(radii <= 0.25)]: ", radii[np.where(radii <= 0.25)])
            raise ValueError("Radii must be above 0.25, but got radii lower")

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


# ---------------------------------------------------------------------------
# Semi-analytical (unbinned/point-process) likelihood machinery
# ---------------------------------------------------------------------------
#
# The classes above are used to *sample* a synthetic catalog, which is needed
# to Monte Carlo integrate the total expected number of detections (the
# completeness function has no closed form, so that integral can't be done
# analytically). The bias problem described for this project comes from a
# different step: putting the *data* into a 5-D histogram before comparing
# it to the model. With ~11 x 14 x 15 x 9 x 8 ~= 1.7e5 voxels and only a few
# thousand independent planets, the overwhelming majority of voxels are
# empty or contain a single planet. The Poisson likelihood of those voxels is
# then dominated by shot noise (does this particular voxel happen to contain
# 0 or 1 planets) rather than by whether the model's *shape* matches the
# data — so a parameter step that makes the 1-D marginals look visibly
# better can easily score worse than one that doesn't, just because it moved
# a couple of points across bin edges in a sparse region.
#
# The functions below implement an unbinned / point-process (a.k.a. extended
# maximum likelihood) formulation instead, of the form used in
# Hogg, Myers & Bovy (2010) and Foreman-Mackey et al. (2014):
#
#   logL = N_obs * log(Gamma0)
#          + sum_j log( <f_pop(theta_j) * completeness(theta_j)>_j )
#          - Gamma0 * Lambda_hat
#
# Gamma0 (the overall occurrence-rate normalization) is NOT sampled by the
# MCMC. For any fixed shape parameters, this is exactly a Poisson-process
# rate-normalization problem, so the logL-maximizing Gamma0 has a closed
# form: Gamma0_opt = N_obs / Lambda_tilde, where Lambda_tilde = Lambda_hat
# evaluated at Gamma0=1 (i.e. the shape-only expected-count integral).
# Substituting Gamma0_opt back in gives the profile log-likelihood actually
# used in kg_likelihood.py:
#   logL_profile = N_obs*log(N_obs) - N_obs*log(Lambda_tilde) - N_obs
#                  + sum_j log( <f_pop(theta_j) * completeness(theta_j)>_j )
# This is valid (not just a convenient approximation) because Gamma0 only
# enters logL through the two terms above, and profiling out a parameter
# with a unique closed-form conditional MLE is exact, not approximate.
# Gamma0's own posterior is not lost by doing this: with this project's
# uniform-in-log10(Gamma0) prior (equivalent to a 1/Gamma0 prior in
# Gamma0-space), the conditional posterior Gamma0 | shape is EXACTLY
# Gamma(shape=N_obs, rate=Lambda_tilde) -- see
# kg_plots.pointprocess_gamma0_posterior_plot, which draws from this
# conjugate distribution using the per-step Lambda_tilde already stored in
# the emcee blobs, recovering Gamma0's full posterior with no MCMC cost.
#
# where:
#   - theta_j = (period, mass, radius, e, omega) for the j-th real detected
#     planet. Each planet's true theta is only known through a posterior of
#     draws (from the photodynamical/TTV fit), so <...>_j denotes an average
#     over that planet's own posterior draws — the standard way of
#     marginalizing per-object measurement uncertainty into a population
#     (hierarchical) likelihood. Critically, this requires *no binning at
#     all*: every posterior draw is evaluated at its own exact location.
#   - f_pop is evaluated in closed form (the *_log_pdf functions below).
#     Period, mass, and eccentricity were already analytic; radius-given-mass
#     is analytic too — it's a one-sided-truncated Gaussian with mean/width
#     from RadiusDistribution.mu_total/sigma_total (matching how
#     sample_radius_given_mass actually draws radii). This is the "radius
#     isn't directly findable" case: the *marginal* p(R) has no closed form
#     (it would require integrating over the mass distribution through a
#     mass-dependent truncation), but we don't need the marginal, since every
#     data point already has its own observed mass to condition on.
#   - completeness/transit-probability still have to come from the
#     precomputed/interpolated grids in RPMeoGrid — there's no way around a
#     numerical piece there — so this is a genuine analytic + numerical
#     hybrid, evaluated pointwise rather than through any histogram.
#   - Lambda_hat is the Monte Carlo estimate of
#     N_stars * E_theta~f_pop[p_det(theta) * p_tr(theta)], from the same
#     synthetic-catalog machinery used elsewhere in this module. That term
#     was already just a sum/mean underneath the old 5-D histogram (the
#     histogram didn't change its value, just made it slower and coupled it
#     to a grid), so it's left as a direct Monte Carlo sum here.
#   - Important subtlety (Neil & Rogers 2020, correcting Foreman-Mackey et al.
#     2014): p_det (pipeline detection efficiency) and p_tr (geometric transit
#     probability) are NOT interchangeable. p_det belongs only in the
#     Lambda_hat integral; including it a second time in the per-planet sum
#     would double-condition on the detection of a planet that's already
#     confirmed to be in the catalog (Loredo 2004; Mandel et al. 2019). p_tr
#     legitimately belongs in both places, since only the transiting subset
#     of the population is ever observable at all. So kg_likelihood.py's
#     Lambda_hat uses RPMeoGrid.interpolate_completeness (p_det*p_tr), while
#     the per-planet data term uses RPMeoGrid.interpolate_transit_probability
#     (p_tr alone) -- see parametric_log_likelihood_pointprocess.
# ---------------------------------------------------------------------------


def profile_optimal_gamma0(n_events, lambda_tilde):
    """
    Closed-form logL-maximizing Gamma0 given the shape parameters, where
    lambda_tilde is the shape-only (Gamma0=1) expected-count integral
    (Lambda_hat/Lambda_hat_grid with Gamma0 divided back out). See the
    "Semi-analytical (unbinned/point-process) likelihood machinery" comment
    block above for the derivation. Shared by kg_likelihood.py's pointprocess
    and grid likelihoods, and by kg_plots.py's region-of-interest rate/
    posterior-reconstruction helpers, so there's exactly one place this
    formula lives.
    """
    if not np.isfinite(lambda_tilde) or lambda_tilde <= 0:
        return np.nan
    return n_events / lambda_tilde


def _powerlaw_segment_integral(beta, P_lo, P_hi):
    """Closed-form integral of P**beta dP from P_lo to P_hi."""
    if np.isclose(beta, -1.0, atol=1e-6):
        return np.log(P_hi / P_lo)
    return (P_hi ** (beta + 1.0) - P_lo ** (beta + 1.0)) / (beta + 1.0)


def period_log_pdf(P, beta1, beta2, period_break_1, P_min=0.1, P_max=500.0):
    """
    Analytic (grid-free) log-density of the broken power-law period
    distribution, matching the piecewise shape used by
    PeriodDistribution/Period_pdf (power_laws=2) exactly, but normalized
    with a closed-form integral instead of a fine-grid trapezoid. There is
    no grid resolution to tune and no dependence on how many points that
    grid happens to have.
    """
    P = np.asarray(P, dtype=np.float64)

    I1 = _powerlaw_segment_integral(beta1, P_min, period_break_1)
    I2 = period_break_1 ** (beta1 - beta2) * _powerlaw_segment_integral(beta2, period_break_1, P_max)
    Z = I1 + I2

    log_shape = np.where(
        P <= period_break_1,
        beta1 * np.log(P),
        (beta1 - beta2) * np.log(period_break_1) + beta2 * np.log(P),
    )
    logpdf = log_shape - np.log(Z)
    return np.where((P >= P_min) & (P <= P_max), logpdf, -np.inf)


def mass_log_pdf(M, mu_M, sigma_M):
    """Analytic log-normal mass density (already closed form, no grid needed)."""
    return lognorm.logpdf(M, s=sigma_M, scale=np.exp(mu_M))


def radius_given_mass_log_pdf(R, M, γ0, γ1, γ2, mass_break_1, mass_break_2, σ0, σ1, σ2, C, density_upper_limit=10.0):
    """
    Analytic conditional log-density of radius given mass: a Gaussian with
    mean/width from RadiusDistribution.mu_total/sigma_total, truncated below
    at the radius implied by density_upper_limit (matching
    RadiusDistribution.sample_radius_given_mass, so this likelihood is
    self-consistent with how the synthetic catalog is actually generated).
    """
    M = np.asarray(M, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    radius_dist = RadiusDistribution(γ0, γ1, γ2, mass_break_1, mass_break_2, σ0, σ1, σ2, C)
    S1 = radius_dist._SN(M, mass_break_1)
    S2 = radius_dist._SN(M, mass_break_2)
    mu = radius_dist.mu_total(M, S1, S2)
    sigma = mu * radius_dist.sigma_total(M, S1, S2)

    lower_bound = radius_given_density_mass(density_upper_limit, M)
    a = (lower_bound - mu) / sigma
    z = (R - mu) / sigma

    logpdf = norm.logpdf(z) - np.log(sigma) - norm.logsf(a)
    return np.where(R >= lower_bound, logpdf, -np.inf)


def eccentricity_log_pdf(e, alpha, lam, sigma_e):
    """
    Analytic eccentricity density. rayleigh_exponential is already a
    properly normalized PDF over e in [0,1] (the two mixture components are
    each normalized to the [0,1] truncation), so this just wraps it in
    log-space.
    """
    e = np.asarray(e, dtype=np.float64)
    pdf = EccentricityDistribution(np.array([0.0, 1.0], dtype=np.float32), alpha, lam, sigma_e).rayleigh_exponential(e)
    return np.log(np.maximum(pdf, 1e-300))


def eccentricity_log_pdf_gamma_mixture(e, mu1, alpha1, mu2, alpha2, f, e_max=0.99):
    """
    Analytic log-density for a 2-component Gamma mixture on eccentricity,
    parametrized by each component's MEAN eccentricity (mu = alpha/beta)
    and shape (alpha), rather than raw (alpha, rate) directly -- beta is
    recovered internally as alpha/mu. f is the mixing weight on component 1
    (f=1 -> pure component 1, f=0 -> pure component 2), same "weight on the
    first term" convention as alpha in eccentricity_log_pdf above.

    Why (mu, alpha) instead of (alpha, beta): mu is directly "what
    eccentricity does this component center on" -- easy to bound with a
    physically meaningful prior. alpha alone then controls how peaked vs.
    diffuse the component is around that mean (coefficient of variation =
    1/sqrt(alpha)), independent of where the mean sits. Under raw
    (alpha, beta), those two roles are tangled: alpha can grow arbitrarily
    large with beta scaled to match (holding the mean fixed) and the
    component silently collapses into an arbitrarily narrow spike, with no
    prior on beta alone actually preventing it. That's the exact same
    soft-max/log-mean-exp collapse already diagnosed for this project's
    Rayleigh+Exponential sigma_e -- (mu, alpha) lets kg_priors.py put a
    direct, bounded prior on alpha to stop it from recurring here.

    kg_priors.py also enforces mu1 < mu2 (component 1 = "tight/low-e",
    component 2 = "broad/higher-e") via an ordering constraint in
    kg_likelihood.parametric_log_prior. Without that, the two components
    are only identifiable up to a label swap (the standard mixture-model
    non-identifiability problem) -- the MCMC would be equally happy calling
    the tight component "1" or "2" from step to step, which shows up as
    spuriously multimodal marginal posteriors on (mu1, alpha1) vs.
    (mu2, alpha2) even though the fitted density itself is unimodal-stable.

    A raw Gamma distribution lives on (0, inf), not [0, 1], so each
    component is truncated to [0, e_max] and renormalized by its own CDF
    at e_max (gammainc(alpha, beta*e_max) is already the REGULARIZED lower
    incomplete gamma function, i.e. exactly the Gamma CDF, so dividing by
    it directly rescales area-under-the-curve back to 1 over [0, e_max]).
    This plays the same role the (1 - exp(...)) denominators play for
    rayleigh_exponential's truncation to [0, 1] -- skip it and the two
    components aren't proper densities over the actual domain, which
    silently penalizes/rewards parameter choices based on how much tail
    mass they lose past e_max rather than on genuine shape fit.

    Computed entirely in log-space (gammaln for the normalizing constant,
    logsumexp for combining the two weighted components) rather than via
    direct pdf evaluation -- e**(alpha-1) and beta**alpha can overflow or
    underflow for realistic shape/rate values, especially for shape < 1
    (which gives an integrable singularity at e=0, not a numerical error,
    but only if handled in log-space).
    """
    e = np.asarray(e, dtype=np.float64)

    if mu1 <= 0 or mu2 <= 0 or alpha1 <= 0 or alpha2 <= 0 or f < 0 or f > 1:
        # Should never happen if kg_priors.py's bounds are respected, but the
        # MCMC can still propose an out-of-bounds step before the prior
        # rejects it -- fail safe with -inf rather than nan/crash from
        # dividing by a non-positive mu or taking gammaln of a non-positive
        # alpha (matching the domain-guard style of the other *_log_pdf
        # functions above, e.g. radius_given_mass_log_pdf's lower_bound mask).
        return np.full(e.shape, -np.inf)

    beta1 = alpha1 / mu1
    beta2 = alpha2 / mu2

    e_safe = np.maximum(e, 1e-12)  # avoid log(0); real data should never be exactly 0 anyway

    log_pdf1_full = alpha1 * np.log(beta1) + (alpha1 - 1) * np.log(e_safe) - beta1 * e_safe - gammaln(alpha1)
    log_pdf2_full = alpha2 * np.log(beta2) + (alpha2 - 1) * np.log(e_safe) - beta2 * e_safe - gammaln(alpha2)

    log_pdf1 = log_pdf1_full - np.log(gammainc(alpha1, beta1 * e_max))
    log_pdf2 = log_pdf2_full - np.log(gammainc(alpha2, beta2 * e_max))

    with np.errstate(divide="ignore"):  # log(f) or log(1-f) at the f=0/1 boundary is a legitimate -inf, not an error
        log_terms = np.stack([np.log(f) + log_pdf1, np.log(1.0 - f) + log_pdf2])
    logpdf = logsumexp(log_terms, axis=0)
    return np.where((e >= 0) & (e <= e_max), logpdf, -np.inf)


def omega_log_pdf(omega, low=0.0, high=360.0):
    """Omega is modeled as uniform, so its density is a constant; included so
    the joint density below is a properly normalized 5-D density."""
    omega = np.asarray(omega, dtype=np.float64)
    logpdf = np.full(omega.shape, -np.log(high - low))
    return np.where((omega >= low) & (omega <= high), logpdf, -np.inf)


def joint_log_intrinsic_density(params, P, M, R, e, omega,model_id=0):
    """
    Fully analytic, grid-free evaluation of the intrinsic population density
    f_pop(period, mass, radius, e, omega | params) at specific (real or
    synthetic) points. This is the core piece of the point-process
    likelihood: instead of putting data into a 5-D histogram and comparing
    binned counts, every point (e.g. every posterior draw of every real
    planet) gets its own exact density evaluation, so the shape of the
    model directly determines the likelihood contribution of every point.

    Parameter unpacking matches get_probability_distributions exactly.
    Note: Gamma0 (overall rate normalization) is not one of these params --
    it's profiled out analytically in kg_likelihood.py rather than sampled,
    so `params` here only ever contains the shape parameters (17 of them).
    """
    γ0, γ1, γ2 = params[0], params[1], params[2]
    σ0, σ1, σ2 = params[3], params[4], params[5]
    mass_break_1, mass_break_2 = params[6], params[7]
    C = params[8]
    mu_M, sigma_M = params[9], params[10]
    β1, β2 = params[11], params[12]
    Period_break_1 = params[13]
    if model_id == 0:
        α, λ, σ_e = params[14], params[15], params[16]
        log_f = (
            period_log_pdf(P, β1, β2, Period_break_1)
            + mass_log_pdf(M, mu_M, sigma_M)
            + radius_given_mass_log_pdf(R, M, γ0, γ1, γ2, mass_break_1, mass_break_2, σ0, σ1, σ2, C)
            + eccentricity_log_pdf(e, α, λ, σ_e)
            + omega_log_pdf(omega)
            )
    elif model_id == 1:
        # Order here (mu_e_1, mu_e_2, alpha_e_1, alpha_e_2, f) matches the
        # adjacent mu_e_1/mu_e_2 ordering in kg_priors.py -- that adjacency is
        # what lets parametric_log_prior enforce mu_e_1 < mu_e_2 (params[i] vs
        # params[i+1]) the same way it already does for Mbreak1/Mbreak2.
        mu1e, mu2e, α1e, α2e, f = params[14], params[15], params[16], params[17], params[18]
        log_f = (
            period_log_pdf(P, β1, β2, Period_break_1)
            + mass_log_pdf(M, mu_M, sigma_M)
            + radius_given_mass_log_pdf(R, M, γ0, γ1, γ2, mass_break_1, mass_break_2, σ0, σ1, σ2, C)
            + eccentricity_log_pdf_gamma_mixture(e, mu1e, α1e, mu2e, α2e, f)
            + omega_log_pdf(omega)
        )

    return log_f


def load_flat_observed_catalog(csv_path):
    """
    Loads the flat (unbinned) KDC catalog written by kg_initialize_voxel_grid.py
    (final_kdc_df / final_kdc.csv) for use by the point-process likelihood.
    Every row is one posterior draw of one real planet; grouping by
    'unique_planet' and averaging within a group is how per-planet
    measurement uncertainty gets marginalized out (see
    joint_log_intrinsic_density and
    kg_likelihood.parametric_log_likelihood_pointprocess).

    Rows are sorted by group so per-step evaluation can use
    np.add.reduceat / np.maximum.reduceat for a fully vectorized grouped
    log-sum-exp, with no Python-level looping over planets. This is a
    one-time (load-time) cost -- like stellar_info, it should be loaded once
    in kg_run_param.py and broadcast to all ranks, not reloaded per step.

    Returns a dict of numpy arrays: P, M, R, e, omega (float64, sorted by
    group), seg_starts (int64 start index of each planet's block within
    those arrays), seg_counts (int64 number of draws for that planet), and
    n_planets.
    """
    cols = ["Period_days", "M_pE", "R_pE", "e", "omega", "unique_planet"]
    df = pd.read_csv(csv_path, usecols=cols, engine="pyarrow")
    df = df.dropna(subset=cols)

    # Sort by group so that each planet's draws form one contiguous block --
    # this is what makes the reduceat-based grouped log-sum-exp possible.
    df = df.sort_values("unique_planet", kind="mergesort").reset_index(drop=True)

    group_ids = df["unique_planet"].to_numpy()
    _, seg_starts, seg_counts = np.unique(group_ids, return_index=True, return_counts=True)
    # group_ids is already sorted, so np.unique's first-occurrence indices
    # land exactly on each block's start position within the sorted arrays.

    return {
        "P": df["Period_days"].to_numpy(dtype=np.float64),
        "M": df["M_pE"].to_numpy(dtype=np.float64),
        "R": df["R_pE"].to_numpy(dtype=np.float64),
        "e": df["e"].to_numpy(dtype=np.float64),
        "omega": df["omega"].to_numpy(dtype=np.float64),
        "seg_starts": seg_starts.astype(np.int64),
        "seg_counts": seg_counts.astype(np.int64),
        "n_planets": len(seg_counts),
    }


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
        assert not np.isnan(cpdds).any() , "CDPP values should not be NaN"
        assert not np.isinf(cpdds).any() , "CDPP values should not be infinite"
        
        durations = [1.5,2,2.5,3,3.5,4.5,5,6,7.5,9,10.5,12,12.5,15]
        assert len(cpdds) == len(durations), "There should be 14 CDPP values corresponding to the durations."

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cdpp_val = cdpp_f(transit_duration)
        return cdpp_val if not np.isnan(cdpp_val) else cdpp_model(transit_duration, A, B)
    
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


def generate_catalog(stellar_info,get_probability_distributions_return,rank,master_seed=None,time_seed=10):

    return_dict = get_probability_distributions_return
    variables = return_dict["variables"]
    
    # np.random.seed(22)


    # print("begin generating fake catalog...")
    # print("len_stellar_df: ",len(stellar_df))

    begin_time = time.time()

    fake_catalog = np.zeros(((len_stellar_info:=len(stellar_info)),5)) # times 10 to test effects of undersampling
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

    # print("generation init time: ", (init_time:=time.time()) - begin_time)


    fake_catalog[:,0] = rng.choice(return_dict["Period_fine_grid"],size=len_stellar_info,p=return_dict["pmf_Period"])  # Period

    # print("period gen time: ", (period_gen_time:=time.time()) - init_time)


    fake_catalog[:,1] = rng.choice(return_dict["mass_fine_grid"],size=len_stellar_info,p=return_dict["pmf_mass"])  # Mass
    mask = fake_catalog[:,1] < 0.1
    while np.any(mask):
        print("Some masses are less than 0.1 M_E, regenerating...")
        fake_catalog[:,1][mask] = rng.choice(return_dict["mass_fine_grid"],size=len(fake_catalog[:,1][mask]),p=return_dict["pmf_mass"])

    # print("number of mass 4 - 24: ", np.sum((fake_catalog[:,1] > 4) & (fake_catalog[:,1] < 24)))
    
    # print("mass gen time: ", (mass_gen_time:=time.time()) - period_gen_time)

    
    # print("number of M greater than 5000: ", np.sum(fake_catalog[:,1]>5000))    
    
    # print("make radius distribution...")
    fake_catalog[:,2] = RadiusDistribution(variables["γ0"],variables["γ1"],variables["γ2"],variables["mass_break_1"],
                                           variables["mass_break_2"],variables["σ0"],variables["σ1"],variables["σ2"],
                                          variables["C"]).sample_radius_given_mass(fake_catalog[:,1],rng)  # Radius
    # fake_catalog[:,2] = np.random.choice(fake_catalog[:,1],size=len_stellar_info,p=p_radius)  # Radius THIS NEEDS EDITING RADIUS IS WEIRD
    # print("radius gen time: ", (radius_gen_time:=time.time()) - mass_gen_time)
    
    fake_catalog[:,3] = rng.choice(return_dict["eccentricity_fine_grid"],size=len_stellar_info,p=return_dict["pmf_ecc"])  # Eccentricity

    # print("ecc gen time: ", (ecc_gen_time:=time.time()) - radius_gen_time)

    fake_catalog[:,4] = rng.uniform(0,360,len_stellar_info)  # omega (argument of periastron)
    # fake_catalog[:,5] = np.random.uniform(-1,1,len_stellar_df)  # b (impact parameter) ... do we need this? why do we need it?
    
    # print("omega gen time: ", (omega_gen_time:=time.time()) - ecc_gen_time)

    return fake_catalog, rng_metadata


def params_to_variables_dict(params, model_id=0):
    """
    Unpack the 17-vector of sampled parameters into a dict of named
    variables, for use by get_probability_distributions and
    joint_log_intrinsic_density. This is a convenience wrapper to avoid
    repeating the same unpacking code in multiple places.
    """
    γ0 = params[0]
    γ1 = params[1]
    γ2 = params[2]
    σ0 = params[3]
    σ1 = params[4]
    σ2 = params[5]
    mass_break_1 = params[6]
    mass_break_2 = params[7]
    C = params[8]
    mu_M = params[9]
    sigma_M = params[10]
    β1 = params[11]
    β2 = params[12]
    Period_break_1 = params[13]

    if model_id == 0:
        α, λ, σ_e = params[14], params[15], params[16]
        return {
            "γ0": γ0,
            "γ1": γ1,
            "γ2": γ2,
            "σ0": σ0,
            "σ1": σ1,
            "σ2": σ2,
            "mass_break_1": mass_break_1,
            "mass_break_2": mass_break_2,
            "C": C,
            "mu_M": mu_M,
            "sigma_M": sigma_M,
            "β1": β1,
            "β2": β2,
            "Period_break_1": Period_break_1,
            "α": α,
            "λ": λ,
            "σ_e": σ_e
        }
    elif model_id == 1:
        mu_e_1, mu_e_2, α_e_1, α_e_2, f = params[14], params[15], params[16], params[17], params[18]
        return {
            "γ0": γ0,
            "γ1": γ1,
            "γ2": γ2,
            "σ0": σ0,
            "σ1": σ1,
            "σ2": σ2,
            "mass_break_1": mass_break_1,
            "mass_break_2": mass_break_2,
            "C": C,
            "mu_M": mu_M,
            "sigma_M": sigma_M,
            "β1": β1,
            "β2": β2,
            "Period_break_1": Period_break_1,
            "mu_e_1": mu_e_1,
            "mu_e_2": mu_e_2,
            "α_e_1": α_e_1,
            "α_e_2": α_e_2,
            "f": f
        }


def get_probability_distributions(params, model_id=0):
    """
    model_id selects which eccentricity model params[14:] holds -- 0 is the
    original 3-parameter Rayleigh+Exponential (params[14:17] = alpha, lambda,
    sigma_e), 1 is the 2-component Gamma mixture (params[14:19] =
    mu_e_1, mu_e_2, alpha_e_1, alpha_e_2, f). Everything before index 14 is
    shared across model_id and unpacked identically. This mirrors
    joint_log_intrinsic_density's model_id branch exactly -- same parameter
    count/order/meaning per model_id -- since get_probability_distributions
    and joint_log_intrinsic_density need to agree about what `params` means
    to build a self-consistent synthetic catalog vs. data-term likelihood
    for the same fitted model. Defaults to model_id=0 for callers that
    predate this argument (e.g. any script still passing a bare 17-vector).
    """
    # unpack params (Gamma0 is not among these -- it's profiled out
    # analytically in kg_likelihood.py rather than sampled; see
    # profile_optimal_gamma0 and the point-process comment block above)
    variables = params_to_variables_dict(params, model_id)


    # period
    Period_fine_grid = np.linspace(0.1,500,10000,dtype=np.float32)
    pdf_Period = PeriodDistribution(Period_fine_grid,[variables["β1"],variables["β2"]],[variables["Period_break_1"]],power_laws=2).Period_pdf(Period_fine_grid)
    pmf_Period = normalize_pdf_to_pmf(pdf_Period,Period_fine_grid)
    # p_Period = normalize_pdf_to_pmf(pdf_Period, Period_fine_grid)

    # mass
    mass_fine_grid = np.logspace(-1,4,10000,dtype=np.float32) # used to be np.linspace(.1,10000,100000) that might be right?
    pdf_mass = MassDistribution(mass_fine_grid,variables["mu_M"],variables["sigma_M"]).mass_pdf()
    pmf_mass = normalize_pdf_to_pmf(pdf_mass,mass_fine_grid)

    # print("pmass: ", p_mass)
    # print("area under mass distribution: ", np.trapezoid(pdf_mass, mass_fine_grid))

    # radius

    # ecc
    eccentricity_fine_grid = np.linspace(0,1,10000,dtype=np.float32)
    if model_id == 1:
        pdf_ecc = np.exp(eccentricity_log_pdf_gamma_mixture(eccentricity_fine_grid, variables["mu_e_1"], variables["α_e_1"], variables["mu_e_2"], variables["α_e_2"], variables["f"]))
    else:
        pdf_ecc = EccentricityDistribution(eccentricity_fine_grid,variables["α"],variables["λ"],variables["σ_e"]).eccentricity_pdf(eccentricity_fine_grid)
    pmf_ecc = normalize_pdf_to_pmf(pdf_ecc, eccentricity_fine_grid)
    # print("p_ecc: ", p_ecc)
    # print("area under eccentricity distribution: ", np.trapezoid(p_ecc, eccentricity_grid))


    is_nan_in_pmfs = (np.isnan(pmf_ecc).any() or np.isnan(pmf_Period).any() or np.isnan(pmf_mass).any())
    #     # print("Warning: PMFs contain NaN. This parameter draw is bad, let's skip it!")

    is_inf_in_pmfs = (not np.isfinite(pmf_ecc).any() or not np.isfinite(pmf_Period).any() or not np.isfinite(pmf_mass).any())
    #     # print("Warning: PMFs contain inf. This parameter draw is bad, let's skip it!")

    is_neg_in_pmfs = (np.any(pmf_ecc < 0) or np.any(pmf_Period < 0) or np.any(pmf_mass < 0))
        # print("Warning: PMFs contain negative values. This parameter draw is bad, let's skip it!")

    get_probability_distributions_return_dict = {"variables": variables,
                                                "pmf_Period": pmf_Period,
                                                "Period_fine_grid": Period_fine_grid,
                                                "pmf_mass": pmf_mass,
                                                "mass_fine_grid": mass_fine_grid,
                                                "pmf_ecc": pmf_ecc,
                                                "eccentricity_fine_grid": eccentricity_fine_grid,
                                                "bad_draw_flags": {
                                                    "is_nan_in_pmfs": is_nan_in_pmfs,
                                                    "is_inf_in_pmfs": is_inf_in_pmfs,
                                                    "is_neg_in_pmfs": is_neg_in_pmfs
                                                    }
                                                } 

    return get_probability_distributions_return_dict


def normalize_pdf_to_pmf(pdf, grid):
    dx = np.gradient(grid)
    pmf = pdf * dx
    pmf /= np.sum(pmf)
    return pmf


def synthetic_catalog_to_grid(synthetic_catalog, voxel_grid, stellar_info, synthetic_multiplier):
    # print("synthetic catalog head: ", synthetic_catalog[:5,:])
    # print("synthetic catalog count: ", np.sum(synthetic_catalog))
    # originally synthetic catalog is in order period, mass, radius, ecc, omega confirmed 12/19 that this is working right
    synthetic_catalog, stellar_info = synthetic_catalog_rearrange_trim(synthetic_catalog,voxel_grid,stellar_info)

    synthetic_catalog = synthetic_catalog_remove_implausible(synthetic_catalog,stellar_info)
    
    ### need to implement a "realistic" filter
    ### wherein planets that go too close to their star (and possibly do anything else unphysical) are removed

    ### RegGridInterp is 5x or worse slower than map_interp, and gives identical results, we're removing it
    # before_reg_interp_time = time.time()

    before_map_interp_time = time.time()
    completeness = voxel_grid.interpolate_completeness(synthetic_catalog)
    # print("map interp time: ", (after_map_interp_time:=time.time()) - before_map_interp_time)


    # print("max(abs(diff)) for map vs RegGridInterp methods :", np.max(np.abs(completeness - completeness_alt)))
    
    ## the max abs diff for the two is 10e-15, seems like these methods are essentially the same


    # print("completeness shape: ", completeness.shape)
    # print("completeness head: ", completeness[:5])
    # print("completeness min/max/mean:", completeness.min(), completeness.max(), completeness.mean())
    # print("completeness near-zero fraction:", np.mean(completeness < 0.1))
    # print("completeness above-one fraction:", np.mean(completeness > 1))
    # print("completeness nan count:", np.sum(np.isnan(completeness)))

    bins = [
        voxel_grid.radius_grid_array,
        voxel_grid.period_grid_array,
        voxel_grid.mass_grid_array,
        voxel_grid.eccentricity_grid_array,
        voxel_grid.omega_grid_array
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

def synthetic_catalog_with_weights(synthetic_catalog,voxel_grid,stellar_info):
    synthetic_catalog, stellar_info = synthetic_catalog_rearrange_trim(synthetic_catalog,voxel_grid,stellar_info)
    
    completeness = voxel_grid.interpolate_completeness(synthetic_catalog)
    
    return synthetic_catalog, completeness, stellar_info


def synthetic_catalog_rearrange_trim(synthetic_catalog,voxel_grid,stellar_info):
    synthetic_catalog = synthetic_catalog[:, [2, 0, 1, 3, 4]]

    # print("rearranged catalog: ", synthetic_catalog)    
    mask = (
        (synthetic_catalog[:, 0] >= np.min(voxel_grid.radius_grid_array)) &
        (synthetic_catalog[:, 0] <= np.max(voxel_grid.radius_grid_array)) &
        (synthetic_catalog[:, 1] >= np.min(voxel_grid.period_grid_array)) &
        (synthetic_catalog[:, 1] <= np.max(voxel_grid.period_grid_array)) &
        (synthetic_catalog[:, 2] >= np.min(voxel_grid.mass_grid_array)) &
        (synthetic_catalog[:, 2] <= np.max(voxel_grid.mass_grid_array)) &
        (synthetic_catalog[:, 3] >= np.min(voxel_grid.eccentricity_grid_array)) &
        (synthetic_catalog[:, 3] <= np.max(voxel_grid.eccentricity_grid_array)) &
        (synthetic_catalog[:, 4] >= np.min(voxel_grid.omega_grid_array)) &
        (synthetic_catalog[:, 4] <= np.max(voxel_grid.omega_grid_array))
    )
    return synthetic_catalog[mask], stellar_info[mask]


def synthetic_catalog_remove_implausible(synthetic_catalog,stellar_info):
    '''Operates on the rearranged synthetic catalog and removes any planets that come within 2 stellar radii of their star.'''
    sm_axis = (synthetic_catalog[:,1] * 24 * 60 * 60)**(2/3) * (G / (4*np.pi**2))**(1/3) * (synthetic_catalog[:,2] * MEKG + stellar_info[:,1] * MSKG)**(1/3)
    periapsis = (1 - synthetic_catalog[:,3]) * sm_axis
    # print("Number of excluded implausible planets: ", np.sum(~(periapsis >= 2 * stellar_info[:,0])))
    synthetic_catalog = synthetic_catalog[periapsis >= 2 * stellar_info[:,0]]
    return synthetic_catalog
