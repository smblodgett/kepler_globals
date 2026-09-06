''''''

import numpy as np

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


def find_earth_flux(P, e, omega Rstar, Tstar, Mstar):
    """
    Calculate the Earth flux received by a planet based on its orbital period and stellar radius.

    Returns:
        float: The Earth flux in units of Earth's flux.
    """
    return   # Example value in units of Earth's flux


def mass_loss_timescale(Menv, Rprim, F_XUV,Fp,eps):
    """
    Calculate the mass loss timescale for a planet due to photoevaporation.

    Returns:
        float: The mass loss timescale in years.
    """
    G = 6.67430e-11  # Gravitational constant

    return G * Menv**2 / (np.pi * eps * Rprim**3 * F_XUV * Fp)  # Example value in years


def p_retention(a, tloss, tau):
    """
    Calculate the retention probability of a planet's atmosphere.

    Returns:
        float: The retention probability (between 0 and 1).
    """
    return np.min(a * tloss / tau , np.ones(a.shape),axis=1)  # Example value between 0 and 1