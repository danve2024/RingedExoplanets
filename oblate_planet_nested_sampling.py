from nested_sampling import NestedSampler, fixed_exoplanet_sma, fixed_exoplanet_mass, fixed_star_object, fixed_pixel_size
from alternative_models import oblate_planet
from units import *

param_bounds = {
    'exoplanet_orbit_eccentricity': (0, 0.4),
    'exoplanet_orbit_inclination': (0, 90),
    'exoplanet_longitude_of_ascending_node': (90 - 5e-8, 90 + 5e-8),
    'exoplanet_argument_of_periapsis': (0, 180),
    'exoplanet_radius': (2000 * km, 20 * 6400 * km),
    'oblateness': (0, 1),
    'rotation': (0, 180)
}

def call(params):
    best_fit_model_output, transit_duration, exoplanet_obj = oblate_planet(
            fixed_exoplanet_sma,
            params[0],  # exoplanet_orbit_eccentricity
            params[1],  # exoplanet_orbit_inclination
            params[2],  # exoplanet_longitude_of_ascending_node
            params[3],  # exoplanet_argument_of_periapsis
            params[4],  # exoplanet_radius
            fixed_exoplanet_mass,
            params[5],  # oblateness
            params[6],  # rotation
            fixed_star_object,
            fixed_pixel_size,
            custom_units=False
        )
    return best_fit_model_output, transit_duration, exoplanet_obj

nlive = 500
ndim = 7

# Initialize and run the nested sampler
oblate_planet_sampler = NestedSampler(
    nlive=nlive,
    ndim=ndim,
    files={
        "C18 short cadence": ("observations/C18_short_cadence.csv", "red"),
        "C18 long cadence": ("observations/C18_long_cadence.csv", "green"),
        "C5": ("observations/C5.csv", "blue"),
    },
    labels=[
        'Orbit Eccentricity',
        'Orbit Inclination, °',
        'Ex. Long. Asc. Node, °',
        'Ex. Arg. of Periapsis, °',
        'Exoplanet Radius, km',
        'Exoplanet Oblateness',
        'Ex. Projection Rotation, °'
    ],
    loglike_file='loglikes/oblate_planet.json',
    model_fn=call,
    parameter_boundaries=param_bounds,
    dynamic_boundaries=False,
    bootstrap=0
)

if __name__ == "__main__":
    results = oblate_planet_sampler.run()
    oblate_planet_sampler.save('alternative_model_runs/oblate_planet/nested_sampling_results.npz')
    oblate_planet_sampler.analyze('alternative_model_runs/oblate_planet/')