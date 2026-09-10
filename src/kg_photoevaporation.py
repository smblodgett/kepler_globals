import numpy as np

from kg_constants import G, MEKG, MSKG, RECM

# Nominal photoevaporation constants (Neil & Rogers 2020, Sec 3.2). `a`
# (their alpha) is the single free scaling parameter that absorbs
# uncertainty in these three nominal values -- see p_retention below.
NOMINAL_EPS = 0.1  # mass-loss efficiency
NOMINAL_F_XUV_E100 = 0.504  # W/m^2 == 504 erg/s/cm^2, XUV flux at Earth-equivalent distance at 100 Myr
NOMINAL_TAU_YR = 5.0e9  # years (5 Gyr), nominal stellar age
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def find_envelope_mass(M, scaling_break=20):
    """
    Calculate the envelope mass of a planet based on its total mass.

    Returns:
        float: The envelope mass in Earth masses.
    """
    def scaling_function(M, scaling_break):
        return 1 / (1 + np.exp(-5 * (np.log(M) - np.log(scaling_break))))
    return 0.1 * M * (1 - scaling_function(M, scaling_break)) + scaling_function(M, scaling_break) * (M - np.sqrt(M))


def find_primordial_radii(M, rng):
    """
    Sample a primordial (still-gaseous) radius for each mass, from the
    same gaseous mass-radius relation used for the "currently gaseous"
    branch, using NR20's own quoted central prior values. `rng` must be
    the caller's already-seeded np.random.Generator, so this draw stays
    reproducible given a fixed seed rather than depending on global numpy
    random state.
    """
    # Local import to avoid a circular import: kg_probability_distributions
    # imports p_retention (etc.) from this module at ITS top level, so this
    # module cannot ALSO import from kg_probability_distributions at its
    # own top level -- Python would try to pull RadiusDistribution off a
    # not-yet-finished kg_probability_distributions module and fail with
    # "cannot import name ... from partially initialized module". Deferring
    # this import to call time (long after both modules have finished
    # loading) breaks the cycle.
    from kg_probability_distributions import RadiusDistribution

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


def find_earth_flux(P, M, e, omega, inc, Rstar, Tstar, Mstar, rng):
    """
    Calculate the incident bolometric flux on a planet, relative to
    Earth's (i.e. F_p/F_Earth, dimensionless) -- Neil & Rogers (2020)
    Sec 3.2's F_p, used to scale F_XUV,E100 up to the planet's own orbit.

    Units: P in days, M in Earth masses, e unitless, omega/inc in degrees,
    Rstar in solar radii, Tstar in Kelvin, Mstar in solar masses. `rng` is
    the caller's seeded np.random.Generator (used for Earth's own,
    otherwise-unconstrained argument of periapsis) so this stays
    reproducible instead of depending on global numpy random state.
    """
    TSUN = 5778
    EEARTH = 0.0167  # Eccentricity of Earth's orbit
    AEARTH = 1.496e11  # Semi-major axis of Earth's orbit in meters

    # (Rstar/Rsun)^2 -- Rstar is already in solar radii, so no extra
    # division by a solar radius is needed here (the original divided by
    # RSUN**2 a second time, which was only correct if Rstar were already
    # in meters -- it wasn't, given every caller has Rstar in solar radii).
    stellar_stuff = Rstar**2 * Tstar**4 / TSUN**4

    omega_rad = omega * np.pi / 180
    inc_rad = inc * np.pi / 180
    # true_anomaly's leading (pi/2 - omega) term previously subtracted
    # omega directly, as if already in radians, while the cos/sin(omega)
    # terms in the very same expression converted degrees->radians -- an
    # internal unit mismatch. omega/inc are degrees everywhere else in
    # this codebase (e.g. kg_probability_distributions.omega_log_pdf), so
    # the leading term is fixed to convert too.
    true_anomaly = np.pi / 2 - omega_rad - e * np.cos(omega_rad) * np.cos(inc_rad) / (1 + e * np.sin(omega_rad))

    P_sec = P * 24 * 3600
    M_kg = M * MEKG
    Mstar_kg = Mstar * MSKG
    dist_p = (G * P_sec**2 * (Mstar_kg + M_kg) / (4 * np.pi**2))**(1/3) * (1 - e**2) / (1 + e * np.cos(true_anomaly))

    earth_omega = rng.uniform(0, 360, size=np.shape(P))  # Randomly sample Earth's argument of periapsis
    earth_omega_rad = earth_omega * np.pi / 180
    earth_true_anomaly = np.pi / 2 - earth_omega_rad - EEARTH * np.cos(earth_omega_rad) * np.cos(inc_rad) / (1 + EEARTH * np.sin(earth_omega_rad))
    dist_e = AEARTH * (1 - EEARTH**2) / (1 + EEARTH * np.cos(earth_true_anomaly))

    return stellar_stuff * dist_e**2 / dist_p**2


def mass_loss_timescale(Menv, Rprim, F_XUV, Fp, eps):
    """
    Calculate the mass loss timescale for a planet due to photoevaporation
    (Lopez et al. 2012 hydrodynamic-escape scaling, as used in Neil &
    Rogers 2020 Sec 3.2). Menv in Earth masses, Rprim in Earth radii,
    F_XUV in W/m^2 (already at the planet's Earth-equivalent reference
    distance), Fp dimensionless (planet's incident flux relative to
    Earth's -- see find_earth_flux), eps dimensionless mass-loss
    efficiency. Menv/Rprim are converted to SI internally so the whole
    expression is dimensionally a timescale.

    Returns:
        float: The mass loss timescale in years.
    """
    Menv_kg = Menv * MEKG
    Rprim_m = Rprim * (RECM / 100)
    t_loss_seconds = G * Menv_kg**2 / (np.pi * eps * Rprim_m**3 * F_XUV * Fp)
    return t_loss_seconds / SECONDS_PER_YEAR


def p_retention(a, tloss, tau):
    """
    Calculate the retention probability of a planet's atmosphere.
    Neil & Rogers (2020) Eq. 14: p_ret = min(a * t_loss/tau, 1), clipped
    below at 0 for safety (a, tloss and tau should all be non-negative,
    but this keeps the result a valid probability even if a proposed
    step momentarily pushes one of them out of range).

    Returns:
        float: The retention probability (between 0 and 1).
    """
    return np.clip(a * tloss / tau, 0.0, 1.0)


def find_mass_loss_timescale(M, R, P, e, omega, inc, Rstar, Mstar, Tstar, rng):
    """
    Calculate the mass loss timescale for a planet due to photoevaporation,
    from natural catalog units: M in Earth masses, R in Earth radii
    (currently unused -- kept for interface symmetry with the other
    per-planet quantities), P in days, e unitless, omega/inc in degrees,
    Rstar in solar radii, Mstar in solar masses, Tstar in Kelvin. `rng`
    must be the caller's already-seeded np.random.Generator; it's threaded
    through to find_primordial_radii and find_earth_flux so that every
    stochastic piece of this calculation is reproducible given a fixed
    seed, instead of each drawing from a fresh, unseeded RNG per call.

    Returns:
        float: The mass loss timescale in years.
    """
    Menv = find_envelope_mass(M)
    Rprim = find_primordial_radii(M, rng)
    F_XUV = NOMINAL_F_XUV_E100
    Fp = find_earth_flux(P, M, e, omega, inc, Rstar, Tstar, Mstar, rng)
    eps = NOMINAL_EPS
    return mass_loss_timescale(Menv, Rprim, F_XUV, Fp, eps)
