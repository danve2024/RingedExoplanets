import math
from typing import Union
from measure import Measure
from units import au, yrs
from scipy.constants import G

# Independent formulas used for model calculation

"""
See the documentation of used parameters and their abbreviations used in comments in the first section of this document: <insert link>. 
"""

def volume(radius) -> Union[float, Measure.Unit]:
    """
    See formula 2.1.1.
    Computes the volume of a sphere.

    :param radius: R
    :return: V
    """
    return 4/3 * math.pi * radius**3

def roche_radius(exoplanet_radius, exoplanet_density, ring_density, roche_constant: float = 2.4423) -> Union[float, Measure.Unit]:
    """
    See formulas 2.1.2 and 2.1.3.
    Computes the Roche radius for an exoplanet.

    :param exoplanet_radius: R
    :param exoplanet_density: D
    :param ring_density: ρ
    :param roche_constant: C_Roche
    :return: d_Roche
    """

    return roche_constant * exoplanet_radius * math.pow(exoplanet_density/ring_density, 1/3)

def hill_sphere(exoplanet_sma, exoplanet_orbit_eccentricity, stellar_mass, exoplanet_mass):
    """
    See formula 2.1.4.
    Completes the Hill radius for the exoplanet.
    
    :param exoplanet_sma: A
    :param exoplanet_orbit_eccentricity: e_p
    :param stellar_mass: M_S
    :param exoplanet_mass: M
    :return: d_Hill
    """
    return exoplanet_sma * math.pow((1 - exoplanet_orbit_eccentricity) * (exoplanet_mass / (3 * stellar_mass)), 1/3)

def roche_sma_min(exoplanet_radius, exoplanet_density, ring_eccentricity, ring_density, roche_constant: float = 2.4423) -> Union[float, Measure.Unit]:
    """
    See formula 2.1.5.
    Computes the minimum semi-major axis of an exoplanet due to Roche radius.

    :param exoplanet_radius: R
    :param exoplanet_density: D
    :param ring_eccentricity: e
    :param ring_density: ρ
    :param roche_constant: C_Roche
    :return: a_Roche_min
    """
    return roche_radius(exoplanet_radius, exoplanet_density, ring_density, roche_constant) / (1 - ring_eccentricity)

def roche_sma_max(exoplanet_radius, exoplanet_density, ring_eccentricity, ring_density, roche_constant: float = 2.4423) -> Union[float, Measure.Unit]:
    """
    See formula 2.1.6.
    Computes the maximum semi-major axis of an exoplanet due to Roche radius.

    :param exoplanet_radius: R
    :param exoplanet_density: D
    :param ring_eccentricity: e
    :param ring_density: ρ
    :param roche_constant: C_Roche
    :return: a_Roche_max
    """
    return roche_radius(exoplanet_radius, exoplanet_density, ring_density, roche_constant) / (1 + ring_eccentricity)

def hill_sma(exoplanet_sma, exoplanet_orbit_eccentricity, ring_eccentricity, stellar_mass, exoplanet_mass) -> Union[float, Measure.Unit]:
    """
    See formula 2.1.7.
    :param exoplanet_sma: A
    :param exoplanet_orbit_eccentricity: e_p
    :param ring_eccentricity: e
    :param stellar_mass: M_S
    :param exoplanet_mass: M
    :return: a_Hill
    """
    return hill_sphere(exoplanet_sma, exoplanet_orbit_eccentricity, stellar_mass, exoplanet_mass) / (1 + ring_eccentricity)

def maximum_ring_mass(exoplanet_mass, exoplanet_radius, ring_sma, ring_eccentricity) -> Union[float, Measure.Unit]:
    """
    See formula 2.1.8.
    Computes the maximum mass of a ring for the exoplanet so that the ring-asteroid system is not binary.

    :param exoplanet_mass: M
    :param exoplanet_radius: R
    :param ring_sma: a
    :param ring_eccentricity: e
    :return: m_max
    """
    return (exoplanet_mass * exoplanet_radius)/(2 * ring_sma * (1 + ring_eccentricity))

def roche_density(exoplanet_mass, ring_sma, ring_eccentricity, roche_density_constant: float = 1.6) -> Union[float, Measure.Unit]:
    """
    See formulas 2.1.9 and 2.1.10.
    Computes the Roche critical density of the ring.

    :param exoplanet_mass: M
    :param ring_sma: a
    :param ring_eccentricity: e
    :param roche_density_constant: C_ρ
    :return: ρ_Roche
    """
    return (3 * exoplanet_mass) / (roche_density_constant * ring_sma ** 3 * (1 + ring_eccentricity) ** 3)

def star_mass(star_log_g, star_radius) -> Union[float, Measure.Unit]:
    """
    See formula 2.1.11.
    Computes the stellar mass based on its log(g) and radius:

    :param star_log_g: log(g)
    :param star_radius: R_S
    :return: M_S
    """
    star_g = 10 ** star_log_g
    return (star_g * star_radius ** 2) / G

def orbital_period(sma, mass_sum) -> Union[float, Measure]:
    """
    See formula 2.1.12.
    Computes the orbital period of a celestial body using the 2nd Kepler's Law.

    :param sma: A
    :param mass_sum: M_S
    :return: P
    """
    return 2 * math.pi * math.sqrt(sma ** 3 / (G * mass_sum))

def mean_anomaly(time, period) -> Union[float, Measure.Unit]:
    """
    See formula 2.1.13.
    Computes the mean anomaly of a body based on the time from its periapsis.

    :param time: t
    :param period: P
    :return: M_A
    """
    return 2 * 360 * time / period

def eccentric_anomaly(_mean_anomaly, eccentricity, tol=1e-10, max_iter=100) -> Union[float, Measure.Unit]:
    """
    See formula 2.1.14.
    Numerically approximates the eccentric anomaly of a body using the Kepler equation for ellipse.

    :param _mean_anomaly: M_A
    :param eccentricity: e_p
    :param tol: float (tolerance for convergence, default=1e-10)
    :param max_iter: int (maximum iterations, default=100)
    :return: E
    """

    _mean_anomaly = math.radians(_mean_anomaly)

    # Newton-Raphson method
    # f(E) = E - esinE - M
    # f'(E) = 1 - ecosE
    # E_(n+1) = E_n - f(E_n)/f'(E_n)

    _eccentric_anomaly = _mean_anomaly

    for _ in range(max_iter):
        delta = (_eccentric_anomaly - eccentricity * math.sin(_eccentric_anomaly) - _mean_anomaly) / (1 - eccentricity * math.cos(_eccentric_anomaly))
        _eccentric_anomaly -= delta
        if abs(delta) < tol:
            return math.degrees(_eccentric_anomaly)
    return math.degrees(_eccentric_anomaly)

def true_anomaly(_eccentric_anomaly, eccentricity) -> Union[float, Measure.Unit]:
    """
    See formula 2.1.15.
    Computes true anomaly based on eccentricity and eccentric anomaly.

    :param _eccentric_anomaly: E
    :param eccentricity: e
    :return: ν
    """
    numerator = math.sqrt(1 + eccentricity) * math.tan(math.radians(_eccentric_anomaly / 2))
    denominator = math.sqrt(1 - eccentricity)
    _true_anomaly = 2 * math.atan2(numerator, denominator)
    return math.degrees(_true_anomaly)

def radius_vector(sma, eccentricity, _true_anomaly) -> Union[float, Measure.Unit]:
    """
    See formula 2.1.16.
    Computes the distance of an orbiting body to the orbit focus.

    :param sma: A
    :param eccentricity: e_p
    :param _true_anomaly: ν
    :return: r
    """
    return sma * (1 - eccentricity ** 2) / (1 - eccentricity * math.cos(math.radians(_true_anomaly)))

def light_curve(data: list) -> list:
    """
    Formats the data created by simulation for creating a lightcurve Δm(Φ)
    [I1, I2, ..., Ii] -> [(t1, Δm1), (t2, Δm2), ..., (ti, Δmi)]

    See formula 2.1.17.

    :param list data: [(t1, I1), (t2, I2), ..., (ti, Ii)]
    :return: [(t1, Δm1), (t2, Δm2), ..., (ti, Δmi)]
    :rtype: list
    """

    ans = []

    intensities = [i[1] for i in data]

    initial_intensity = data[0][1] # I_min

    for i in range(len(intensities)):
        intensity = intensities[i]
        magnitude_change = 2.5 * math.log10(initial_intensity/intensity)
        ans.append((data[i][0], magnitude_change))

    return ans

def to_pixels(linear_size, pixel = 100_000) -> Union[float, Measure.Unit]:
    """
    See equation 2.1.18.
    Converts linear units to pixel units for creating mask matrices.

    :param linear_size: l
    :param pixel: px
    :return: converted linear unit to pixel unit
    """
    return linear_size / pixel
