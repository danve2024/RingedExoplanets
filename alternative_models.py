from typing import Union, List, Dict, Any, Tuple
import json
import os
import hashlib

import numpy as np
from measure import Measure
from models import quadratic_star_model, square_root_star_model
from space import CustomStarModel, Star, Orbit, Exoplanet, StarSpot
from units import *


def get_parameter_hash(**kwargs) -> str:
    """Generate a unique hash for the given parameters."""
    # Convert all parameters to strings and sort them for consistency
    param_str = json.dumps({k: str(v) for k, v in sorted(kwargs.items())}, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


def load_noise_seeds() -> dict:
    """Load noise seeds from loglikes/noise_seeds.json."""
    seeds_file = 'loglikes/noise_seeds.json'
    if os.path.exists(seeds_file):
        try:
            with open(seeds_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load noise seeds file {seeds_file}: {e}")
            print("Creating backup of corrupted file and starting fresh...")
            # Create backup of corrupted file
            backup_file = f"{seeds_file}.corrupted"
            try:
                import shutil
                shutil.copy2(seeds_file, backup_file)
                print(f"Corrupted file backed up to {backup_file}")
            except Exception as backup_error:
                print(f"Could not create backup: {backup_error}")
            return {}
    return {}


def save_noise_seeds(seeds: dict):
    """Save noise seeds to loglikes/noise_seeds.json."""
    seeds_file = 'loglikes/noise_seeds.json'
    os.makedirs(os.path.dirname(seeds_file), exist_ok=True)
    with open(seeds_file, 'w') as f:
        json.dump(seeds, f, indent=2)


def get_or_create_seed(**params) -> int:
    """Get existing seed for parameters or create a new one."""
    seeds = load_noise_seeds()
    param_hash = get_parameter_hash(**params)
    
    if param_hash not in seeds:
        # Generate new seed from hash
        seeds[param_hash] = int(hashlib.md5(param_hash.encode()).hexdigest()[:8], 16) % (2**31)
        save_noise_seeds(seeds)
    
    return seeds[param_hash]
    
def ringless(exoplanet_sma: Union[float, Measure.Unit], exoplanet_orbit_eccentricity: Union[float, Measure.Unit],
             exoplanet_orbit_inclination: Union[float, Measure.Unit],
             exoplanet_longitude_of_ascending_node: Union[float, Measure.Unit],
             exoplanet_argument_of_periapsis: Union[float, Measure.Unit],
             exoplanet_radius: Union[float, Measure.Unit], exoplanet_mass: Union[float, Measure.Unit],
             star: Union[CustomStarModel, Star], pixel_size: int, custom_units=True, **kwargs) -> tuple:
    """Calculate the simulation data"""
    # Parameters
    if custom_units:
        exoplanet_sma = exoplanet_sma.set(au)
        exoplanet_orbit_inclination = exoplanet_orbit_inclination.set(deg)
        exoplanet_longitude_of_ascending_node = exoplanet_longitude_of_ascending_node.set(deg)
        exoplanet_argument_of_periapsis = exoplanet_argument_of_periapsis.set(deg)
        exoplanet_radius = exoplanet_radius.set(km)
        exoplanet_mass = exoplanet_mass.set(kg)

    stellar_mass = star.mass

    orbit = Orbit(exoplanet_sma, exoplanet_orbit_eccentricity, exoplanet_orbit_inclination,
                  exoplanet_longitude_of_ascending_node, exoplanet_argument_of_periapsis, stellar_mass,
                  pixel_size)  # create exoplanet orbit
    exoplanet = Exoplanet(orbit, exoplanet_radius, exoplanet_mass, pixel=pixel_size)  # create exoplanet

    duration, data = star.transit(exoplanet)

    return data, duration, exoplanet


def spotted_star_transit(sma: Union[float, Measure.Unit], 
                        eccentricity: Union[float, Measure.Unit],
                        inclination: Union[float, Measure.Unit],
                        lan: Union[float, Measure.Unit],
                        aop: Union[float, Measure.Unit],
                        radius: Union[float, Measure.Unit], 
                        mass: Union[float, Measure.Unit],
                        star_radius: Union[float, Measure.Unit],
                        star_log_g: Union[float, Measure.Unit],
                        limb_darkening_coefficients,
                        pixel_size: int,
                        angular_velocity: float = 1.,
                        spot_longitude: float = 0.,
                        spot_radius: float = 0.05,
                        spot_brightness: float = 0.7,
                        limb_darkening_model: str = 'quadratic',
                        **kwargs) -> Tuple[List[Tuple[float, float]], float, Exoplanet]:
    """
    Simulate a planet transiting a star with spots/faculae.
    
    Args:
        sma: Semi-major axis
        eccentricity: Orbital eccentricity
        inclination: Inclination in degrees
        lan: Longitude of ascending node
        aop: Argument of periapsis
        radius: Planet radius
        mass: Planet mass
        star_radius: Star radius
        star_log_g: Logarithm of stellar surface gravity
        limb_darkening_coefficients: limb-darkening coefficients for the quadratic model
        pixel_size: Pixel scale
        angular_velocity: Angular velocity of the spots on the star
        spot_longitude: Longitude of the equatorial spot
        spot_radius: Spot size as fraction of stellar radius
        spot_brightness: Spot contrast (<1 for dark spots, >1 for bright faculae)
        limb_darkening_model: 'quadratic' or 'square-root'
        
    Returns:
        Tuple of (lightcurve_data, duration, exoplanet)
    """
    spot = StarSpot(spot_longitude, spot_radius, spot_brightness)

    match limb_darkening_model:
        case "quadratic":
            star_model = quadratic_star_model
        case "square-root":
            star_model = square_root_star_model
        case _:
            raise ValueError('Invalid star model selected.')

    
    star = CustomStarModel(star_model, star_radius, star_log_g, limb_darkening_coefficients, pixel_size, angular_velocity, spot)

    orbit = Orbit(sma, eccentricity, inclination, lan, aop, star.mass, pixel_size)
    exoplanet = Exoplanet(orbit, radius, mass, pixel=pixel_size)

    duration, data = star.transit(exoplanet)

    return data, duration, exoplanet

def oblate_planet(sma: Union[float, Measure.Unit], eccentricity: Union[float, Measure.Unit],
                   inclination: Union[float, Measure.Unit],
                   lan: Union[float, Measure.Unit],
                   aop: Union[float, Measure.Unit],
                   radius: Union[float, Measure.Unit], mass: Union[float, Measure.Unit],
                   oblateness: Union[float, Measure.Unit], rotation: Union[float, Measure.Unit],
                   star: Union[CustomStarModel, Star], pixel_size: int, custom_units=True, **kwargs) -> tuple:
    """Calculate the simulation data"""
    # Parameters
    if custom_units:
        sma = sma.set(au)
        inclination = inclination.set(deg)
        lan = lan.set(deg)
        aop = aop.set(deg)
        radius = radius.set(km)
        mass = mass.set(kg)
        rotation = rotation.set(deg)

    stellar_mass = star.mass

    orbit = Orbit(sma, eccentricity, inclination, lan, aop, stellar_mass, pixel_size)  # create exoplanet orbit
    exoplanet = Exoplanet(orbit, radius, mass, oblateness=oblateness, rotation_angle=rotation, pixel=pixel_size)  # create exoplanet

    duration, data = star.transit(exoplanet)

    return data, duration, exoplanet


def noise(exoplanet_sma: Union[float, Measure.Unit], exoplanet_orbit_eccentricity: Union[float, Measure.Unit],
               exoplanet_orbit_inclination: Union[float, Measure.Unit],
               exoplanet_longitude_of_ascending_node: Union[float, Measure.Unit],
               exoplanet_argument_of_periapsis: Union[float, Measure.Unit],
               exoplanet_radius: Union[float, Measure.Unit], exoplanet_mass: Union[float, Measure.Unit],
               star: Union[CustomStarModel, Star], pixel_size: int, noise_scale: float, noise_magnitude: float, 
               seed: int = None, custom_units=True, **kwargs) -> tuple:
    """
    Noise model that calls ringless model to get lightcurve, adds noise, and converts back to magnitude.
    
    Args:
        noise_scale: Scale parameter for noise generation (multiplies the random normal values)
        noise_magnitude: Standard deviation multiplier for noise generation  
        seed: Random seed for reproducible noise generation (if None, generates unique seed based on parameters)
        **kwargs: Additional parameters passed to ringless model
        
    Returns:
        Tuple of (noisy_data, duration, exoplanet)
    """
    # Generate or get seed for this parameter set
    if seed is None:
        # Create parameter dictionary for seed generation
        param_dict = {
            'exoplanet_sma': str(exoplanet_sma),
            'exoplanet_orbit_eccentricity': str(exoplanet_orbit_eccentricity),
            'exoplanet_orbit_inclination': str(exoplanet_orbit_inclination),
            'exoplanet_longitude_of_ascending_node': str(exoplanet_longitude_of_ascending_node),
            'exoplanet_argument_of_periapsis': str(exoplanet_argument_of_periapsis),
            'exoplanet_radius': str(exoplanet_radius),
            'exoplanet_mass': str(exoplanet_mass),
            'star': str(star),
            'pixel_size': pixel_size,
            'noise_scale': noise_scale,
            'noise_magnitude': noise_magnitude,
            'custom_units': custom_units,
            **kwargs
        }
        seed = get_or_create_seed(**param_dict)
    
    np.random.seed(seed)

    data, duration, exoplanet = ringless(
        exoplanet_sma, exoplanet_orbit_eccentricity, exoplanet_orbit_inclination,
        exoplanet_longitude_of_ascending_node, exoplanet_argument_of_periapsis,
        exoplanet_radius, exoplanet_mass, star, pixel_size, custom_units, **kwargs
    )

    relative_flux = 10 ** (-0.4 * np.array([point[1] for point in data]))

    noise = np.random.normal(scale=noise_scale, size=relative_flux.shape) * noise_magnitude
    noisy_flux = relative_flux + noise

    noisy_magnitudes = -2.5 * np.log10(noisy_flux)

    noisy_data = [(data[i][0], noisy_magnitudes[i]) for i in range(len(data))]
    
    return noisy_data, duration, exoplanet
