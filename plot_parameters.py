import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def load_and_plot(input_file, n_params=7, n_samples=10000):
    """Load the samples and plot their distributions."""
    try:
        # Load the samples
        samples = np.load(input_file)
        
        # If we have more parameters than needed, take the first n_params
        if samples.shape[1] > n_params:
            samples = samples[:, :n_params]
        
        # Get random subset of samples
        if len(samples) > n_samples:
            idx = np.random.choice(len(samples), size=n_samples, replace=False)
            samples = samples[idx]
        
        # Parameter labels (update these based on your model)
        all_labels = [
            'Orbit Eccentricity',
            'Orbit Inclination, °',
            'Ex. Long. Asc. Node, °',
            'Ex. Arg. of Periapsis, °',
            'Exoplanet Radius, km',
            'Ring Eccentricity',
            'Ring Obliquity, °',
            'Ring Azimuthal Angle, °',
            'Ring Arg. of Periapsis, °',
            'Ring Inner Radius, km',
            'Ring Outer Radius, km'
        ]
        
        labels = all_labels[:n_params]
        
        # Print basic statistics
        print("\nParameter statistics:")
        print("-" * 70)
        for i, label in enumerate(labels):
            param = samples[:, i]
            print(f"{label:<30}: mean={np.mean(param):.3f}, std={np.std(param):.3f}, "
                  f"min={np.min(param):.3f}, max={np.max(param):.3f}")
        
        # Plot histograms for each parameter
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 3 * n_rows))
        
        for i in range(n_params):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(samples[:, i], bins=50, alpha=0.7, density=True)
            plt.title(labels[i])
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parameter_distributions.png')
        print("\nSaved parameter distributions to 'parameter_distributions.png'")
        
        # Save the processed samples
        np.save('processed_samples.npy', samples)
        print("\nSaved processed samples to 'processed_samples.npy'")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot parameter distributions from extracted samples')
    parser.add_argument('--input', '-i', default='extracted_floats/samples_7d.npy',
                      help='Input file with extracted samples (default: extracted_floats/samples_7d.npy)')
    parser.add_argument('--params', '-p', type=int, default=7,
                      help='Number of parameters (default: 7)')
    parser.add_argument('--samples', '-n', type=int, default=10000,
                      help='Number of samples to use (default: 10000)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    load_and_plot(args.input, args.params, args.samples)
