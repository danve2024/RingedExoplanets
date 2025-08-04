from visualization import Selection, LoadFile, Model, app
from measure import Measure
from units import *
import sys

# Runs the main application with given parameter values

"""
See the documentation of used parameters and their abbreviations used in comments in the first section of this document: <insert link>. 
"""

defaults = {
    'exoplanet_sma': Measure(0.1, 20, au, label='A'),
    'exoplanet_orbit_eccentricity': Measure(0, 0.4, label='e_p'),
    'exoplanet_orbit_inclination': Measure(-90, 90, deg, label='i'),
    'exoplanet_longitude_of_ascending_node': Measure(0, 360, deg, label='Ω'),
    'exoplanet_argument_of_periapsis': Measure(0, 360, deg, label='ω'),
    'exoplanet_radius': Measure(2_000, 350_000, km, label='R'),
    'exoplanet_mass': Measure(1e23, 7e27, kg, label='D'),
    'density': Measure(0.01, 0.03, gcm3, label='ρ'), # Dependent slider
    'eccentricity': Measure(0, 0.4, label='e'),
    'sma': Measure(5e5, 5e6, km, label='a'),  # Dependent slider
    'width': Measure(1e4, 1e6, km, label='w'), # Dependent slider
    'mass': Measure(0.1, 10, kg, label='m'), # Dependent slider
    'obliquity': Measure(-90, 90, deg, label='θ'),
    'azimuthal_angle': Measure(0, 360, deg, label='φ'),
    'argument_of_periapsis': Measure(0, 360, deg, label='ψ'),
    'specific_absorption_coefficient': 2.3e-3 * (m**2/g),
    'star_radius': 700_000 * km,
    'star_temperature': 5500 * K,
    'star_log(g)': 3,
    'wavelength': 3437 * angstrom,
    'band': 'u (quadratic)',
    'limb_darkening': 'quadratic',
    'pixel_size': 10000 * km
}

if len(sys.argv) > 1:
    if sys.argv[1] == 'model':
        print('Parameter selection step is skipped.')
        selector = Selection(defaults)
        selector.click()
        window = Model(selector.user_values, None)
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
