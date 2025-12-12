from nested_sampling import NestedSampler, fixed_exoplanet_sma, fixed_exoplanet_mass, fixed_star_object, \
    fixed_pixel_size, fixed_star_radius, fixed_star_log_g, fixed_star_coefficients
from alternative_models import spotted_star_transit
from units import *

min_rotation_period = 0.2 * days
max_angular_velocity = 360 / min_rotation_period

param_bounds = {
    'exoplanet_orbit_eccentricity': (0, 0.4),
    'exoplanet_orbit_inclination': (0, 90),
    'exoplanet_longitude_of_ascending_node': (90 - 5e-8, 90 + 5e-8),
    'exoplanet_argument_of_periapsis': (0, 180),
    'exoplanet_radius': (2000 * km, 9.2 * 6_400 * km),
    'star_angular_velocity': (0, max_angular_velocity),
    'spot_longitude': (0, 360),
    'spot_radius': (0, 1),
    'spot_brightness': (0, 2)
}

def call(params):
    best_fit_model_output, transit_duration, exoplanet_obj = spotted_star_transit(
            fixed_exoplanet_sma,
            params[0],  # exoplanet_orbit_eccentricity
            params[1],  # exoplanet_orbit_inclination
            params[2],  # exoplanet_longitude_of_ascending_node
            params[3],  # exoplanet_argument_of_periapsis
            params[4],  # exoplanet_radius
            fixed_exoplanet_mass,
            fixed_star_radius,
            fixed_star_log_g,
            fixed_star_coefficients,
            fixed_pixel_size,
            params[5], # star_angular_velocity
            params[6], # spot_longitude
            params[7], # spot_radius
            params[8], # spot_brightness
            custom_units=False
        )
    return best_fit_model_output, transit_duration, exoplanet_obj

nlive = 500
ndim = 7

# Initialize and run the nested sampler
star_spots_sampler = NestedSampler(
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
        'Star Ang. Velocity, °/s',
        'Spot Longitude, °',
        'Spot Radius, Star Radii',
        'Spot Intensity, Star Intensities'
    ],
    loglike_file='loglikes/star_spots.json',
    model_fn=call,
    parameter_boundaries=param_bounds,
    dynamic_boundaries=False
)

if __name__ == "__main__":
    results = star_spots_sampler.run()
    star_spots_sampler.save('alternative_model_runs/star_spots/nested_sampling_results.npz')
    star_spots_sampler.analyze('alternative_model_runs/star_spots/')