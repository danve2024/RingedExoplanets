from nested_sampling import NestedSampler, fixed_exoplanet_sma, fixed_exoplanet_mass, fixed_star_object, fixed_pixel_size
from alternative_models import noise
from units import *
import pandas as pd
import numpy as np
import os

def get_noise_boundaries_from_data():
    """Automatically determine noise scale and magnitude boundaries from observation data."""
    observations_dir = 'observations'
    observation_files = [
        'C18_short_cadence.csv',
        'C18_long_cadence.csv', 
        'C5.csv'
    ]
    
    all_stats = []
    
    for file_name in observation_files:
        file_path = os.path.join(observations_dir, file_name)
        if os.path.exists(file_path):
            try:
                data = pd.read_csv(file_path, header=None, names=['phase', 'flux'])
                
                # Convert flux to magnitude for noise analysis
                magnitudes = -2.5 * np.log10(data['flux'])

                noise_std = np.std(magnitudes)

                window_size = min(10, len(magnitudes) // 4)
                rolling_std = magnitudes.rolling(window=window_size, center=True).std().dropna()
                
                all_stats.append({
                    'noise_std': noise_std,
                    'rolling_std_max': rolling_std.max()
                })
            except Exception as e:
                print(f"Warning: Could not analyze {file_name}: {e}")
    
    if not all_stats:
        return 0.01, 0.01

    max_noise_std = max(stats['noise_std'] for stats in all_stats)
    max_rolling_std = max(stats['rolling_std_max'] for stats in all_stats)

    noise_scale_max = max_rolling_std * 3
    noise_magnitude_max = max_noise_std * 2

    noise_scale_max = max(noise_scale_max, 0.001)
    noise_magnitude_max = max(noise_magnitude_max, 0.001)
    
    print(f"Automatically determined noise boundaries:")
    print(f"  noise_scale_max: {noise_scale_max:.6f}")
    print(f"  noise_magnitude_max: {noise_magnitude_max:.6f}")
    
    return noise_scale_max, noise_magnitude_max

noise_scale_max, noise_magnitude_max = get_noise_boundaries_from_data()

param_bounds = {
    'exoplanet_orbit_eccentricity': (0, 0.4),
    'exoplanet_orbit_inclination': (0, 90),
    'exoplanet_longitude_of_ascending_node': (90 - 5e-8, 90 + 5e-8),
    'exoplanet_argument_of_periapsis': (0, 180),
    'exoplanet_radius': (2000 * km, 20 * 6400 * km),
    'noise_scale': (0, noise_scale_max),
    'noise_magnitude': (0, noise_magnitude_max)
}

def call(params):
    best_fit_model_output, transit_duration, exoplanet_obj = noise(
            fixed_exoplanet_sma,
            params[0],  # exoplanet_orbit_eccentricity
            params[1],  # exoplanet_orbit_inclination
            params[2],  # exoplanet_longitude_of_ascending_node
            params[3],  # exoplanet_argument_of_periapsis
            params[4],  # exoplanet_radius
            fixed_exoplanet_mass,
            fixed_star_object,
            fixed_pixel_size,
            params[5], # noise_scale
            params[6], # noise_magnitude
            custom_units=False
        )
    return best_fit_model_output, transit_duration, exoplanet_obj

nlive = 500
ndim = 7

# Initialize and run the nested sampler
noise_sampler = NestedSampler(
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
        'Noise Scale',
        'Noise Magnitude'
    ],
    loglike_file='loglikes/noise.json',
    model_fn=call,
    parameter_boundaries=param_bounds,
    dynamic_boundaries=False,
    bootstrap=0
)

if __name__ == "__main__":
    results = noise_sampler.run()
    noise_sampler.save('alternative_model_runs/noise/nested_sampling_results.npz')
    noise_sampler.analyze('alternative_model_runs/noise/')