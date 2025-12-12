import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict

def analyze_observation_noise(file_path: str) -> Dict:
    """Analyze noise characteristics of observation data."""
    data = pd.read_csv(file_path, header=None, names=['phase', 'flux'])
    
    # Convert flux to magnitude for noise analysis
    magnitudes = -2.5 * np.log10(data['flux'])
    
    # Calculate noise statistics
    noise_std = np.std(magnitudes)
    noise_range = np.max(magnitudes) - np.min(magnitudes)
    
    # Calculate rolling statistics to understand local noise
    window_size = min(10, len(magnitudes) // 4)
    rolling_std = magnitudes.rolling(window=window_size, center=True).std().dropna()
    
    return {
        'noise_std': noise_std,
        'noise_range': noise_range,
        'rolling_std_mean': rolling_std.mean(),
        'rolling_std_max': rolling_std.max(),
        'data_points': len(data)
    }

def determine_noise_boundaries() -> Tuple[float, float]:
    """Determine appropriate noise scale and magnitude boundaries from observation data."""
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
            stats = analyze_observation_noise(file_path)
            all_stats.append(stats)
            print(f"Analysis for {file_name}:")
            print(f"  Noise std: {stats['noise_std']:.6f}")
            print(f"  Noise range: {stats['noise_range']:.6f}")
            print(f"  Rolling std mean: {stats['rolling_std_mean']:.6f}")
            print(f"  Rolling std max: {stats['rolling_std_max']:.6f}")
            print(f"  Data points: {stats['data_points']}")
            print()
    
    if not all_stats:
        # Fallback values if no observation files found
        return 0.01, 0.01
    
    # Calculate conservative boundaries based on observed data
    max_noise_std = max(stats['noise_std'] for stats in all_stats)
    max_rolling_std = max(stats['rolling_std_max'] for stats in all_stats)
    
    # Set noise_scale to cover the range of observed noise variations
    # Use a factor of 3 to allow for reasonable variation
    noise_scale_max = max_rolling_std * 3
    
    # Set noise_magnitude to cover the total observed noise range
    # Use a factor of 2 to allow for reasonable variation
    noise_magnitude_max = max_noise_std * 2
    
    # Ensure minimum reasonable values
    noise_scale_max = max(noise_scale_max, 0.001)
    noise_magnitude_max = max(noise_magnitude_max, 0.001)
    
    print(f"Derived boundaries:")
    print(f"  noise_scale_max: {noise_scale_max:.6f}")
    print(f"  noise_magnitude_max: {noise_magnitude_max:.6f}")
    
    return noise_scale_max, noise_magnitude_max

if __name__ == "__main__":
    determine_noise_boundaries()
