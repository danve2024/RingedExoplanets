import os.path
import json

import visualization
from units import *
from formulas import *
from measure import Measure
from math import exp
from typing import Union
from models import disk, elliptical_ring, transit, quadratic_star_model, square_root_star_model, transit_animation, planet, show_model, _array_to_qimage
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

    def display(self):
        show_model(self.model)

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
    def __init__(self, orbit: Orbit, radius: Measure.Unit, mass: Measure.Unit, rings: Rings = None,
                 oblateness: float = 0.0, rotation_angle: float = 0.0, pixel=100_000):
        """
        Sets exoplanet parameters and creates its numpy mask array representation with its rings.
        
        Args:
            rings: The planet's ring system
            orbit: The planet's orbital parameters
            radius: The planet's equatorial radius
            mass: The planet's mass
            oblateness: Flattening factor (0 = sphere, ~0.3 = highly oblate)
            rotation_angle: Rotation angle in degrees (0-360)
            pixel: Pixel scale in meters per pixel
        """
        # Asteroid parameters
        self.rings = rings  # ring
        self.orbit = orbit  # orbit
        self.radius = radius  # equatorial radius (R)
        self.mass = mass  # mass (M)
        self.oblateness = float(oblateness)  # flattening factor
        self.rotation_angle = float(rotation_angle)  # rotation angle in degrees
        self.volume = volume(self.radius)  # volume (V)
        self.density = self.mass / self.volume  # density (D), see formula 2.3.3
        self.pixel = pixel  # pixel size (px)
        self.polar_radius = self.radius * (1 - self.oblateness)  # polar radius

        self.px_radius = to_pixels(self.radius, self.pixel)  # radius in pixels (χ_R)
        self.px_polar_radius = to_pixels(self.polar_radius, self.pixel)  # polar radius in pixels

        if self.rings is None:
            self.min_roche_sma, self.max_roche_sma, self.maximum_ring_mass, self.apoapsis = 0, 0, 0, 0
            self.crop_factor = to_pixels(self.radius * 20, pixel)
        else:
            # Parameters for limiting the sliders
            self.min_roche_sma = max(
                roche_sma_min(self.radius, self.density, self.rings.eccentricity, self.rings.density),
                self.radius / (1 + self.rings.eccentricity)
            )  # a_Roche_min
            self.max_roche_sma = roche_sma_max(
                self.radius, self.density, self.rings.eccentricity, self.rings.density
            )  # a_Roche_max
            self.maximum_ring_mass = maximum_ring_mass(
                self.mass, self.radius, self.rings.sma, self.rings.eccentricity
            )  # m_max

            # Create exoplanet model with its rings
            self.rings.init(int(round(to_pixels(2 * self.radius))), pixel)
            self.apoapsis = self.rings.sma * (1 + self.rings.eccentricity)  # apoapsis (r_a)
            self.crop_factor = to_pixels(max(2 * self.radius, 2 * self.apoapsis), pixel)  # cropping factor (CF)
        
        # Create planet model
        self.disk = planet(
            radius=self.px_radius,
            size=self.crop_factor,
            fill=np.inf,
            oblateness=self.oblateness,
            rotation_angle=self.rotation_angle
        )
        
        if self.rings is None:
            self.model = self.disk
        else:
            self.rings.adjust(self.crop_factor)
            self.adjust(self.crop_factor + self.rings.px_width * (1 + self.rings.eccentricity))
            self.model = self.disk + self.rings.model

    def adjust(self, size: Union[float, Measure.Unit]) -> None:
        """
        Adjusts the asteroid model size and rings model size.

        :param size: matrix size (used for matrices concatenation)
        """
        self.disk = planet(
            radius=self.px_radius,
            size=size,
            fill=np.inf,
            oblateness=self.oblateness,
            rotation_angle=self.rotation_angle
        )
        self.rings.adjust(size)
        self.model = self.disk + self.rings.model

    def display(self):
        show_model(self.model)

    def hill_sma(self, stellar_mass):
        if self.rings is None:
            return 0
        return hill_sma(self.orbit.sma, self.orbit.eccentricity, self.rings.eccentricity, stellar_mass, self.mass)

    def info(self):
        return f'exoplanet(R:{self.radius/km}km, M:{self.mass/kg}kg, V:{self.volume/m3}m³, D:{self.density/gcm3}g/cm³, r_a:{self.apoapsis/km}km, χ_R:{self.px_radius}px, CF:{self.crop_factor}px)'

    def __str__(self):
        return self.info() + '\n' + str(self.orbit) + '\n' + str(self.rings)

class StarSpot:
    def __init__(self, initial_longitude=0., radius=0., brightness=1.):
        self.longitude = initial_longitude  # λ_0
        self.radius = radius  # ρ
        self.brightness = brightness  # β

class CustomStarModel:
    def __init__(self, model_function, radius: Union[float, Measure.Unit], log_g: Union[float, Measure.Unit], coefficients: Union[list[float], tuple[float]], pixel=100_000, angular_velocity=0, spot: StarSpot = None):
        self.model_fn = model_function
        if len(coefficients) != 2:
            raise ValueError(f'The star model takes only 2 limb-darkening coefficients. You have {len(coefficients)} instead: {coefficients}')
            
        # Store parameters
        self.radius = radius  # radius (R_S)
        self.log_g = log_g    # log(g)
        self.mass = star_mass(self.log_g, self.radius)  # M_S
        self.c1 = coefficients[0]  # u_1
        self.c2 = coefficients[1]  # u_2
        self.coefficients = [self.c1, self.c2]
        self.pixel = pixel
        self.angular_velocity = angular_velocity # ω_spot
        self.spot = spot
        if self.spot is None:
            self.spot = StarSpot()
        
        # Calculate model dimensions - ensure it's large enough and odd-sized
        model_size = max(101, 2 * int(round(to_pixels(self.radius, pixel)))) + 1  # Ensure odd size for symmetry
        
        # Create the star model
        self.base_model = self.model_fn([model_size, model_size], self.coefficients)
        self.model = self.base_model.copy()

        # Calculate total intensity (flux)
        self.intensity = np.sum(self.model)
        
        # Ensure model is properly normalized to [0,1] range
        model_min = np.min(self.model)
        model_max = np.max(self.model)
        if model_max > model_min:  # Avoid division by zero
            self.model = (self.model - model_min) / (model_max - model_min)

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
            steps=steps,
            angular_velocity=self.angular_velocity,
            spot_longitude=self.spot.longitude,
            spot_radius=self.spot.radius,
            spot_brightness=self.spot.brightness
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
            steps=steps,
            angular_velocity=self.angular_velocity,
            spot_longitude=self.spot.longitude,
            spot_radius=self.spot.radius,
            spot_brightness=self.spot.brightness
        )

    def qimage(self):
        return _array_to_qimage(self.model)

    def display(self):
        show_model(self.model)

class Star(CustomStarModel):
    def __init__(self, radius: Union[float, Measure.Unit], temperature: Union[float, Measure.Unit] = 8000, 
                 log_g: Union[float, Measure.Unit] = 4, wavelength: Union[float, Measure.Unit] = 4687, 
                 band: str = 'V', pixel: int = 100_000, limb_darkening_model: str = 'square-root',
                 angular_velocity=0, spot: StarSpot = None) -> None:
        """
        Initialize a star with optional star spots.
        
        Args:
            radius: Stellar radius in meters
            temperature: Effective temperature in Kelvin
            log_g: Surface gravity (log10 of surface gravity in cgs units)
            wavelength: Wavelength in Angstroms for limb darkening
            band: Photometric band (e.g., 'V', 'B', 'R')
            pixel: Pixel scale in meters per pixel
            limb_darkening_model: Type of limb darkening model ('square-root' or 'quadratic')
            angular_velocity: The angular velocity of the star
            spot: A stellar spot
        """
        self.radius = radius  # radius (R_S)
        self.temperature = temperature  # temperature (T_eff)
        self.log_g = log_g  # log(g)
        self.mass = star_mass(self.log_g, self.radius)
        self.wavelength = wavelength
        self.band = band
        self.limb_darkening_model = limb_darkening_model.lower()
        self.pixel = pixel

        try:
            if self.limb_darkening_model == 'quadratic':
                try:
                    c1, c2 = quadratic_limb_darkening[int(self.temperature)][float(self.log_g)][self.band]  # γ_1, γ_2
                except KeyError:
                    raise KeyError(
                        f'No available darkening coefficients for star with parameters T: {self.temperature/K}K '
                        f'log(g): {self.log_g} band: {self.band}'
                    )
                star_model = quadratic_star_model

            elif self.limb_darkening_model == 'square-root':
                try:
                    c1, c2 = square_root_limb_darkening[int(self.temperature)][float(self.log_g)][int(self.wavelength)]  # γ_3, γ_4
                except KeyError:
                    raise KeyError(
                        f'No available darkening coefficients for star with parameters T: {self.temperature/K}K '
                        f'log(g): {self.log_g} wavelength: {self.wavelength/angstrom}Å'
                    )
                star_model = square_root_star_model

            else:
                raise ValueError(f'Unsupported limb darkening model: {self.limb_darkening_model}. '
                               'Use "quadratic" or "square-root".')

        except IndexError:
            raise ValueError(
                f'Invalid star parameters (T: {self.temperature/K}K, log(g): {self.log_g}, '
                f'λ: {self.wavelength/angstrom}Å, band: {self.band}).\n'
            )

        super().__init__(star_model, self.radius, self.log_g, [c1, c2], pixel, angular_velocity, spot)

    def __str__(self):
        return f"star(R_S: {self.radius/km}km, T: {self.temperature/K}K, log_g: {self.log_g}, λ: {self.wavelength}Å, ω_spot: {self.angular_velocity}°/s,  band: {self.band})"


class Void:
    def __init__(self, **kwargs):
        for i in kwargs:
            self.__setattr__(i, kwargs[i])


if __name__ == "__main__":
    # Create a star spot with visible parameters
    spot = StarSpot(radius=0.1, brightness=0.3, initial_longitude=90)
    
    # Create a star with proper parameters (radius in meters, with spot and rotation)
    sun = Star(
        radius=700_000 * km,  # Star radius in meters
        temperature=5500 * K,
        log_g=3.0,
        wavelength=3437 * angstrom,
        band='u',
        pixel=10000 * km,  # Pixel size in meters
        limb_darkening_model='quadratic',
        angular_velocity=0.002,  # Rotation rate in degrees per second
        spot=spot
    )
    
    # Create an orbit that will result in a visible transit
    # Parameters: sma (au), eccentricity, inclination (deg), longitude_of_ascending_node (deg), 
    # argument_of_periapsis (deg), stellar_mass (kg), pixel_size (m)
    # For edge-on transit (inclination=90°), planet should pass through star center
    # But we need px_sma to be small enough that even at maximum projected distance, 
    # the planet is within max_dist_for_transit
    # With max_dist ~211px, let's use sma that gives px_sma ~150px to be safe
    orbit = Orbit(
        sma=0.005 * au,  # Very small - about 750,000 km (just slightly larger than star radius)
        eccentricity=0.0,  # Circular orbit for simpler transit
        inclination=0 * deg,  # Exactly edge-on for perfect transit
        longitude_of_ascending_node=0 * deg,
        argument_of_periapsis=0 * deg,
        mass_sum=sun.mass,
        pixel=10000 * km
    )
    
    # Create a planet without rings for simplicity
    # Make planet larger relative to star for more visible transit
    exoplanet = Exoplanet(
        orbit=orbit,
        radius=100_000 * km,  # Larger planet radius (about 14% of star radius)
        mass=1e26 * kg,  # Planet mass in kg
        rings=None,  # No rings for this demo
        pixel=10000 * km
    )

    # Generate transit frames
    frames = sun.transit_frames(exoplanet, steps=200)
    
    if frames:
        anim = visualization.FramesWindow(frames)
        anim.save_gif('stellar_spots.gif')
        print(f"Successfully created stellar_spots.gif with {len(frames)} frames")
    else:
        print("Error: No frames generated. Check transit parameters.")
        print("Trying with a planet that has rings...")
        
        # Try with rings to see if that helps
        rings = Rings(
            density=0.02 * gcm3,
            eccentricity=0.1,
            sma=200_000 * km,
            width=50_000 * km,
            obliquity=0 * deg,
            azimuthal_angle=0 * deg,
            argument_of_periapsis=0 * deg,
            mass=1 * kg,
            specific_absorption_coefficient=2.3e-3 * (m**2/g)
        )
        planet_with_rings = Exoplanet(
            orbit=orbit,
            radius=100_000 * km,
            mass=1e26 * kg,
            rings=None,
            pixel=10000 * km
        )
        frames = sun.transit_frames(planet_with_rings, steps=200)
        if frames:
            anim = visualization.FramesWindow(frames)
            anim.save_gif('stellar_spots.gif')
            print(f"Successfully created stellar_spots.gif with {len(frames)} frames (with rings)")
        else:
            print("Still no frames even with rings. Check orbit parameters.")