from typing import Union, List, Dict, Any, Tuple
from measure import Measure
from space import CustomStarModel, Star, Orbit, Exoplanet, StarSpot
from units import *

def spotted_star_transit(sma: Union[float, Measure.Unit], 
                        eccentricity: Union[float, Measure.Unit],
                        inclination: Union[float, Measure.Unit],
                        lan: Union[float, Measure.Unit],
                        aop: Union[float, Measure.Unit],
                        radius: Union[float, Measure.Unit], 
                        mass: Union[float, Measure.Unit],
                        star_radius: Union[float, Measure.Unit],
                        star_temperature: float,
                        star_log_g: float,
                        wavelength: float,
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
        star_radius: Stellar radius
        star_temperature: Stellar effective temperature
        star_log_g: Stellar surface gravity
        wavelength: Observation wavelength
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
    
    star = Star(star_radius, star_temperature, star_log_g, wavelength, 
               'V', pixel_size, limb_darkening_model, spot=spot)

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
