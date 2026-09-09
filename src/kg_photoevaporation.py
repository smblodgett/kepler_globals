''''''

import numpy as np

from kg_constants import G, RSCM
from kg_probability_distributions import MassDistribution, RadiusDistribution


def find_envelope_mass(M,scaling_break=20):
    """
    Calculate the envelope mass of a planet based on its total mass.

    Returns:
        float: The envelope mass in Earth masses.
    """
    def scaling_function(M, scaling_break):
        return 1 / (1 + np.exp(-5 * (np.log(M) - np.log(scaling_break))))
    return 0.1 * M * (1 - scaling_function(M, scaling_break)) + scaling_function(M, scaling_break) * (M - np.sqrt(M)) # Example value


def find_primordial_radii(M,rng):
    γ0 = 0.0  # center of prior from NR20...could be better?
    γ1 = 0.6  # center of prior from NR20...could be better?
    γ2 = 0.0  # center of prior from NR20...could be better?
    mass_break_1 = 7.38  # center of prior from NR20...could be better?
    mass_break_2 = 148.4  # center of prior from NR20...could be better?
    σ0 = 0.165  # center of prior from NR20...could be better?
    σ1 = 0.273  # center of prior from NR20...could be better?
    σ2 = 0.1  # center of prior from NR20...could be better?
    C = 2.5  # center of prior from NR20...could be better?

    primordial_radii = RadiusDistribution(γ0,γ1,γ2,mass_break_1,mass_break_2,σ0,σ1,σ2,C).sample_radius_given_mass(M,rng)
    return primordial_radii


def find_earth_flux(P, M, e, omega, inc, Rstar, Tstar, Mstar):
    """
    Calculate the Earth flux received by a planet based on its orbital period and stellar radius.

    Returns:
        float: The Earth flux in units of Earth's flux.
    """
    RSUN = RSCM / 100
    TSUN = 5778
    EEARTH = 0.0167  # Eccentricity of Earth's orbit
    AEARTH = 1.496e11  # Semi-major axis of Earth's orbit in meters

    stellar_stuff = Rstar**2 * Tstar**4 / (RSUN**2 * TSUN**4)
    true_anomaly = np.pi / 2 - omega - e * np.cos(omega*np.pi / 180) * np.cos(inc * np.pi / 180) / (1 + e * np.sin(omega*np.pi / 180)) # Convert omega and inc to radians
    dist_p = (G * P**2 * (Mstar + M) / (4 * np.pi**2))**(1/3) * (1 - e**2) / (1 + e * np.cos(true_anomaly))
    earth_omega = np.random.uniform(0, 360,size=len(P))  # Randomly sample Earth's argument of periapsis
    earth_true_anomaly = np.pi / 2 - omega - EEARTH * np.cos(earth_omega*np.pi / 180) * np.cos(inc * np.pi / 180) / (1 + EEARTH * np.sin(earth_omega*np.pi / 180)) # Convert omega and inc to radians
    dist_e = AEARTH * (1 - EEARTH**2) / (1 + EEARTH * np.cos(earth_true_anomaly))  # Distance of Earth from Sun at the same true anomaly
    return stellar_stuff * dist_e**2 / dist_p**2  # Example value in units of Earth's flux


def mass_loss_timescale(Menv, Rprim, F_XUV,Fp,eps):
    """
    Calculate the mass loss timescale for a planet due to photoevaporation.

    Returns:
        float: The mass loss timescale in years.
    """
    return G * Menv**2 / (np.pi * eps * Rprim**3 * F_XUV * Fp)  # Example value in years


def p_retention(a, tloss, tau):
    """
    Calculate the retention probability of a planet's atmosphere.

    Returns:
        float: The retention probability (between 0 and 1).
    """
    return np.min(a * tloss / tau , np.ones(a.shape),axis=1)  # Example value between 0 and 1


def find_mass_loss_timescale(M, R, P, e, omega, inc, Rstar, Mstar, Tstar):
    """
    Calculate the mass loss timescale for a planet due to photoevaporation.

    Returns:
        float: The mass loss timescale in years.
    """
    Menv = find_envelope_mass(M)
    Rprim = find_primordial_radii(M, np.random.default_rng())
    F_XUV = 0.504 # sci
    Fp = find_earth_flux(P, M, e, omega, inc, Rstar, Mstar, Tstar) # sci
    eps = 0.1
    return mass_loss_timescale(Menv, Rprim, F_XUV, Fp, eps)