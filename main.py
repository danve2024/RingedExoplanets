from visualization import Selection, LoadFile, Model, app
from measure import Measure
from units import *
from typing import Union
from space import Star, Orbit, Exoplanet, Rings, CustomStarModel
import sys

# Runs the main application with given parameter values

"""
See the documentation of used parameters and their abbreviations used in comments in the first section of this document: <insert link>. 
"""

defaults = {
    'exoplanet_sma': Measure(0.1, 20, au, label='A'),
    'exoplanet_orbit_eccentricity': Measure(0, 0.9, label='e_p'),
    'exoplanet_orbit_inclination': Measure(0, 90, deg, label='i'),
    'exoplanet_longitude_of_ascending_node': Measure(0, 360, deg, label='Ω'),
    'exoplanet_argument_of_periapsis': Measure(0, 360, deg, label='ω'),
    'exoplanet_radius': Measure(2_000, 350_000, km, label='R'),
    'exoplanet_mass': Measure(1e23, 7e27, kg, label='D'),
    'density': Measure(0.01, 0.03, gcm3, label='ρ'), # Dependent slider
    'eccentricity': Measure(0, 0.4, label='e'),
    'sma': Measure(5e5, 5e6, km, label='a'),  # Dependent slider
    'width': Measure(1e4, 1e6, km, label='w'), # Dependent slider
    'mass': Measure(0.1, 10, kg, label='m'), # Dependent slider
    'obliquity': Measure(0, 90, deg, label='θ'),
    'azimuthal_angle': Measure(0, 360, deg, label='φ'),
    'argument_of_periapsis': Measure(0, 360, deg, label='ψ'),
    'specific_absorption_coefficient': 2.3e-3 * (m**2/g),
    'star_radius': 700_000 * km,
    'star_temperature': 5500 * K,
    'star_log(g)': 3.00,
    'wavelength': 3437 * angstrom,
    'band': 'u (quadratic)',
    'limb_darkening': 'quadratic',
    'pixel_size': 10000 * km
}


def calculate_data(exoplanet_sma: Union[float, Measure.Unit], exoplanet_orbit_eccentricity: Union[float, Measure.Unit],
                   exoplanet_orbit_inclination: Union[float, Measure.Unit],
                   exoplanet_longitude_of_ascending_node: Union[float, Measure.Unit],
                   exoplanet_argument_of_periapsis: Union[float, Measure.Unit],
                   exoplanet_radius: Union[float, Measure.Unit], exoplanet_mass: Union[float, Measure.Unit],
                   density: Union[float, Measure.Unit], eccentricity: Union[float, Measure.Unit],
                   sma: Union[float, Measure.Unit], width: Union[float, Measure.Unit], mass: Union[float, Measure.Unit],
                   obliquity: Union[float, Measure.Unit], azimuthal_angle: Union[float, Measure.Unit],
                   argument_of_periapsis: Union[float, Measure.Unit], specific_absorption_coefficient: float,
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
        density = density.set(gcm3)
        sma = sma.set(km)
        width = width.set(km)
        mass = mass.set(kg)
        obliquity = obliquity.set(deg)
        azimuthal_angle = azimuthal_angle.set(deg)
        argument_of_periapsis = argument_of_periapsis.set(deg)

    stellar_mass = star.mass

    rings = Rings(density, eccentricity, sma, width, obliquity, azimuthal_angle, argument_of_periapsis, mass,
                  specific_absorption_coefficient)  # create rings
    orbit = Orbit(exoplanet_sma, exoplanet_orbit_eccentricity, exoplanet_orbit_inclination,
                  exoplanet_longitude_of_ascending_node, exoplanet_argument_of_periapsis, stellar_mass,
                  pixel_size)  # create exoplanet orbit
    exoplanet = Exoplanet(orbit, exoplanet_radius, exoplanet_mass, rings=rings, pixel=pixel_size)  # create exoplanet

    duration, data = star.transit(exoplanet)

    return data, duration, exoplanet

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'model':
            print('Parameter selection step is skipped.')
            selector = Selection(defaults)
            selector.click()
            window = Model(selector.user_values, calculate_data, None)
            window.show()
            sys.exit(app.exec())
    else:
        # Graphics
        selector = Selection(defaults)
        load_file = LoadFile()
        app.exec()
        parameters = selector.user_values
        filename = load_file.filename

        window = Model(parameters, filename)
        window.show()
        sys.exit(app.exec())
