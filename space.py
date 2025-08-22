import os.path
import json

from units import *
from formulas import *
from measure import Measure
from math import exp
from typing import Union
from models import disk, elliptical_ring, transit, quadratic_star_model, square_root_star_model, transit_animation
import numpy as np

# Functions that require other Solar System objects

"""
See the documentation of used parameters and their abbreviations used in comments in the first section of this document: <insert link>. 
"""

def convert_keys_to_int(d):
    ans = {}
    for i in d:
        ans[int(i)] = {}
        for j in d[i]:
            ans[int(i)][float(j)] = {}
            for k in d[i][j]:
                if list(d[i][j].keys())[0].isalpha():
                    ans[int(i)][float(j)][k] = d[i][j][k]
                else:
                    ans[int(i)][float(j)][int(k)] = d[i][j][k]
    return ans

loaded_data = {}

for limb_darkening in ['square-root.json', 'quadratic.json']:
    with open(os.path.join('limb_darkening', limb_darkening), 'r') as f:
        loaded_data[limb_darkening] = convert_keys_to_int(json.load(f))

# Parameters for a square-root limb darkening star model (Diaz-Cordoves, J., & Gimenez, A. (1992). A new nonlinear approximation to the limb-darkening of hot stars.)
# T -> log(g) -> λ
square_root_limb_darkening = loaded_data['square-root.json']

# Parameters for a quadratic limb darkening star model (Claret, A., & Gimenez, A. (1990). Limb-darkening coefficients of late-type stars.)
# T -> log(g) -> Band
quadratic_limb_darkening = loaded_data['quadratic.json']

# Classes of the main celestial bodies used for the model
class Rings:
    """
    A class for the rings model.
    """
    def __init__(self, density: Measure.Unit, eccentricity: Measure.Unit, sma: Measure.Unit, width: Measure.Unit, obliquity: Measure.Unit, azimuthal_angle: Measure.Unit, argument_of_periapsis: Measure.Unit, mass: Measure.Unit, specific_absorption_coefficient=2.3e-3 * (m**2/g)):
        # Ring parameters
        self.density = density # density (ρ)
        self.eccentricity = eccentricity # eccentricity (e)
        self.sma = sma # semi-major axis (a)
        self.width = width # width (w)
        self.obliquity = obliquity # obliquity (θ)
        self.azimuthal_angle = azimuthal_angle # azimuthal angle (φ)
        self.argument_of_periapsis = argument_of_periapsis # argument of periapsis (ψ)
        self.mass = mass # mass (m)

        # For calculating the ring transparency
        self.specific_absorption_coefficient = specific_absorption_coefficient # mass-specific absorption coefficient for silicate dust with quartz dominating (τ_μ)
        self.absorption_coefficient = self.specific_absorption_coefficient * self.density # absorption coefficient for silicate dust, τ (see formula 2.3.1)
        self.absorption = exp(-self.absorption_coefficient) # light absorption (see formula 2.3.2)

        # Parameters defined in self.init()
        self.size = None  # matrix size (n, k)
        self.px_sma = None # semi-major axis in pixels (α)
        self.px_width = None # width in pixels (χ_w)
        self.model = None  # model of the ring

    def init(self, size: int, pixel=100_000) -> None:
        """
        Initializes the ring numpy mask array model using models.py/elliptical_ring

        :param size: χ_a
        :param pixel: px
        """
        self.size = int(round(size)) # matrix size
        self.px_sma = to_pixels(self.sma, pixel)
        self.px_width = to_pixels(self.width, pixel)
        self.model = elliptical_ring(self.size, self.px_sma, self.eccentricity, self.px_width, self.obliquity, self.azimuthal_angle, self.argument_of_periapsis, self.absorption_coefficient)

    def adjust(self, size) -> None:
        """
        Adjusts the ring model size.

        :param size: matrix size (used for matrices concatenation)
        """
        self.model = elliptical_ring(int(round(size)), self.px_sma, self.eccentricity, self.px_width, self.obliquity, self.azimuthal_angle, self.argument_of_periapsis, self.absorption_coefficient)

    def __str__(self) -> str:
        return f'rings(d:{self.density/gcm3}g/cm³, e:{self.eccentricity}, a:{self.sma/km}km, w:{self.width/km}km, θ:{self.obliquity/deg}°, φ:{self.azimuthal_angle/deg}°, ψ:{self.argument_of_periapsis/deg}°, m:{self.mass/kg}kg, κ:{self.specific_absorption_coefficient/(m ** 2 / g)}m²/g, τ:{self.absorption_coefficient}, α:{self.px_sma}px, χ_w:{self.px_width}px)'

class Orbit:
    """
    A class for modeling the orbits of celestial bodies.
    """
    def __init__(self, sma: Measure.Unit, eccentricity: Measure.Unit, inclination: Measure.Unit, longitude_of_ascending_node: Measure.Unit, argument_of_periapsis: Measure.Unit, mass_sum = Measure.Unit, pixel=100_000):
        self.sma = sma # semi-major axis (A)
        self.eccentricity = eccentricity # eccentricity (e_p)
        self.inclination = inclination # inclination (i)
        self.lan = longitude_of_ascending_node # longitude of ascending node (Ω)
        self.argument_of_periapsis = argument_of_periapsis # argument of periapsis (ω)
        self.mass_sum = mass_sum # mass sum, for this case the star mass (M_S)
        self.period = orbital_period(self.sma, self.mass_sum) # orbital period (P)

        self.px_sma = to_pixels(sma, pixel) # semi-major axis in pixels (α_p)


    def __str__(self):
        return f'orbit(a:{self.sma/au}au, e:{self.eccentricity}, i:{self.inclination/deg}°, Ω:{self.lan/deg}°, ω:{self.argument_of_periapsis/deg}°, α_p:{self.px_sma}px)'

class Exoplanet:
    def __init__(self, rings: Rings, orbit: Orbit, radius: Measure.Unit, mass: Measure.Unit, pixel=100_000):
        """
        Sets exoplanet parameters and creates its numpy mask array representation with its rings.
        """
        # Asteroid parameters
        self.rings = rings # ring
        self.orbit = orbit # orbit
        self.radius = radius # radius (R)
        self.mass = mass # mass (M)
        self.volume = volume(self.radius) # volume (V)
        self.density = self.mass / self.volume # density (D), see formula 2.3.3
        self.pixel = pixel # pixel size (px)

        self.px_radius = to_pixels(self.radius, self.pixel) # radius in pixels (χ_R)

        # Parameters for limiting the sliders
        self.min_roche_sma = max(roche_sma_min(self.radius, self.density, self.rings.eccentricity, self.rings.density), self.radius / (1 + self.rings.eccentricity)) # a_Roche_min
        self.max_roche_sma = roche_sma_max(self.radius, self.density, self.rings.eccentricity, self.rings.density) # a_Roche_max
        self.maximum_ring_mass = maximum_ring_mass(self.mass, self.radius, self.rings.sma, self.rings.eccentricity) # m_max

        # Create exoplanet model with its rings
        self.rings.init(int(round(to_pixels(2 * self.radius))), pixel)
        self.apoapsis = self.rings.sma * (1 + self.rings.eccentricity) # apoapsis (r_a), see formula 2.3.4
        self.crop_factor = to_pixels(max(2 * self.radius, 2 * self.apoapsis), pixel) # cropping factor (CF), see formula 2.3.5
        self.disk = disk(to_pixels(self.radius), self.crop_factor) # disk
        self.rings.adjust(self.crop_factor)
        self.adjust(self.crop_factor + self.rings.px_width * (1 + self.rings.eccentricity))
        self.model = self.disk + self.rings.model

    def adjust(self, size: Union[float, Measure.Unit]) -> None:
        """
        Adjusts the asteroid model size and rings model size.

        :param size: matrix size (used for matrices concatenation)
        """
        self.disk = disk(to_pixels(self.radius, self.pixel), size)
        self.rings.adjust(size)
        self.model = self.disk + self.rings.model

    def hill_sma(self, stellar_mass):
        return hill_sma(self.orbit.sma, self.orbit.eccentricity, self.rings.eccentricity, stellar_mass, self.mass)

    def info(self):
        return f'exoplanet(R:{self.radius/km}km, M:{self.mass/kg}kg, V:{self.volume/m3}m³, D:{self.density/gcm3}g/cm³, r_a:{self.apoapsis/km}km, χ_R:{self.px_radius}px, CF:{self.crop_factor}px)'

    def __str__(self):
        return self.info() + '\n' + str(self.orbit) + '\n' + str(self.rings)

class CustomStarModel:
    def __init__(self, model_function, radius: Union[float, Measure.Unit], log_g: Union[float, Measure.Unit], coefficients: Union[list[float], tuple[float]], pixel=100_000):
        self.model_fn = model_function
        if len(coefficients) != 2:
            raise ValueError(f'The star model takes only 2 limb-darkening coefficient. You have {len(coefficients)} instead: {coefficients}')
        self.radius = radius # radius (R_S)
        self.log_g = log_g  # log(g)
        self.mass = star_mass(self.log_g, self.radius) # M_S
        self.c1 = coefficients[0] # u_1
        self.c2 = coefficients[1] # u_2
        self.coefficients = [self.c1, self.c2]
        self.intensity = np.sum(self.model_fn([round(to_pixels(self.radius*2, pixel)), round(to_pixels(self.radius*2, pixel))], self.coefficients))
        self.model = self.model_fn([to_pixels(self.radius * 2, pixel), round(to_pixels(self.radius * 2, pixel))],
                                self.coefficients)

    def __str__(self):
        return f"star(R_S: {self.radius/km}km)"

    def transit(self, exoplanet: Exoplanet, steps: int=500):
        """
        Calculates the transit time and the lightcurve of the magnitude change of the exoplanet transit.

        :param Exoplanet exoplanet: the transiting exoplanet
        :param steps: number of transit frames
        :return: [P, Δm(t)] - exoplanet period and transit light curve
        """

        return transit(
            star=self.model,
            mask=exoplanet.model,
            period=exoplanet.orbit.period,
            eccentricity=exoplanet.orbit.eccentricity,
            sma=exoplanet.orbit.px_sma,
            inclination=exoplanet.orbit.inclination,
            longitude_of_ascending_node=exoplanet.orbit.lan,
            argument_of_periapsis=exoplanet.orbit.argument_of_periapsis,
            steps=steps
        )

    def transit_frames(self, exoplanet: Exoplanet, steps: int = 500):
        """
        Returns the transit frames for an animation window.

        :param Exoplanet exoplanet: the transiting exoplanet
        :param steps: number of transit frames
        :return: [P, Δm(t)] - exoplanet period and transit light curve
        """

        return transit_animation(
            star=self.model,
            mask=exoplanet.model,
            period=exoplanet.orbit.period,
            eccentricity=exoplanet.orbit.eccentricity,
            sma=exoplanet.orbit.px_sma,
            inclination=exoplanet.orbit.inclination,
            longitude_of_ascending_node=exoplanet.orbit.lan,
            argument_of_periapsis=exoplanet.orbit.argument_of_periapsis,
            steps=steps
        )

class Star(CustomStarModel):
    def __init__(self, radius: Union[float, Measure.Unit], temperature: Union[float, Measure.Unit] = 8000, log_g: Union[float, Measure.Unit] = 4, wavelength: Union[float, Measure.Unit] = 4687, band='V', pixel=100_000, limb_darkening_model: str = 'square-root') -> None:
        self.radius = radius # radius (R_S)
        self.temperature = temperature # temperature
        self.log_g = log_g # log(g)
        self.mass = star_mass(self.log_g, self.radius)
        self.wavelength = wavelength
        self.band = band
        self.limb_darkening_model = limb_darkening_model # limb-darkening model


        try:
            if self.limb_darkening_model == 'quadratic':
                try:
                    c1, c2 = quadratic_limb_darkening[int(self.temperature)][float(self.log_g)][self.band] # quadratic limb-darkening coefficients (γ_1, γ_2)
                except KeyError:
                    raise KeyError(f'No available darkening coefficients for star with parameters T: {self.temperature/K}K log(g): {self.log_g} band: {self.band}')
                star_model = quadratic_star_model

            elif self.limb_darkening_model == 'square-root':
                try:
                    c1, c2 = square_root_limb_darkening[int(self.temperature)][float(self.log_g)][int(self.wavelength)] # square-root limb-darkening coefficients (γ_3, γ_4)
                except KeyError:
                    raise KeyError(f'No available darkening coefficients for star with parameters T: {self.temperature/K}K log(g): {self.log_g} wavelength: {self.wavelength/angstrom}Å')
                star_model = square_root_star_model

            else:
                raise ValueError(f'Wrong limb darkening model selected: ' + self.limb_darkening_model)

        except IndexError:
            raise ValueError(f'Wrong star parameter value selected (temperature [{self.temperature/K}K], log_g [{self.log_g}], wavelength [{self.wavelength}] or band [{self.band}]).\nTo see available parameter values check /limb_darkening.')

        super().__init__(star_model, self.radius, [c1, c2], pixel)


    def __str__(self):
        return f"star(R_S: {self.radius/km}km, T: {self.temperature/K}K, log_g: {self.log_g}, λ: {self.wavelength}Å, band: {self.band})"


class Void:
    def __init__(self, **kwargs):
        for i in kwargs:
            self.__setattr__(i, kwargs[i])
