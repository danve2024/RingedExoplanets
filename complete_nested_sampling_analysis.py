import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import corner
import sys
from typing import Union, List, Dict, Tuple
from dynesty import utils as dyfunc
from formulas import roche_sma_max, roche_density, volume
from main import calculate_data, defaults
from models import quadratic_star_model
from units import *
from space import CustomStarModel
import os
import json
import random
from collections import OrderedDict

# Fixed parameters (from the original code)
fixed_pixel_size = defaults['pixel_size']
fixed_star_radius = 1.28 * sun_radius  # Grouffal et al., 2022
fixed_star_log_g = 4.3  # from TICv8
fixed_star_coefficients = [0.0678, 0.188]  # Grant & Wakeford, 2024
fixed_observations_duration = 80 * hrs
fixed_star_object = CustomStarModel(quadratic_star_model, fixed_star_radius,
                                    fixed_star_log_g, fixed_star_coefficients,
                                    fixed_pixel_size)
fixed_exoplanet_mass = 12 * earth_mass  # Santerne et al., 2019
fixed_exoplanet_period = 542.08 * days  # Santerne et al., 2019
fixed_exoplanet_sma = 1.377 * au
fixed_specific_absorption_coefficient = defaults['specific_absorption_coefficient']

min_radius =  2000 * km
max_radius = 9.2 * 6_400 * km
max_semi_major_axis_max = roche_sma_max(max_radius, fixed_exoplanet_mass/volume(min_radius), 0, roche_density(fixed_exoplanet_mass, max_radius, 0.9))
max_width_max = max_semi_major_axis_max - min_radius

specific_parameter_boundaries = {
    'exoplanet_orbit_eccentricity': (0, 0.4),
    'exoplanet_orbit_inclination': (0, 90),
    'exoplanet_longitude_of_ascending_node': (90 - 5e-8, 90 + 5e-8),
    'exoplanet_argument_of_periapsis': (0, 180),
    'exoplanet_radius': (min_radius, max_radius),
    'eccentricity': (0, 0.9),
    'semi_major_axis': (min_radius, max_semi_major_axis_max), # dynamic boundaries
    'width': (0, max_width_max), # dynamic boundaries
    'obliquity': (0, 90),
    'azimuthal_angle': (0, 180),
    'argument_of_periapsis': (0, 180)
}

class NestedSamplingAnalysis:
    def __init__(self, results_file: str = "nested_sampling_results.npz"):
        """
        Initialize the analysis with saved nested sampling results.
        
        Args:
            results_file: Path to the saved nested sampling results
        """
        print("Loading nested sampling results...")
        self.results_file = results_file
        self.load_results()
        self.load_observations()
        
        # Calculate prior standard deviation for uniform distributions
        # For uniform distribution U(a,b), the standard deviation is (b-a)/sqrt(12)
        param_bounds = list(specific_parameter_boundaries.values())
        self.prior_stdev = np.array([(high - low) / np.sqrt(12) for low, high in param_bounds])
        
    def load_results(self):
        """Load the nested sampling results from file."""
        data = np.load(self.results_file)
        self.samples = data['samples']
        self.logwt = data['logwt']
        self.logz = data['logz']
        self.logl = data['logl']
        
        # Create proper labels for all 11 parameters based on the original nested sampling code
        self.labels = [
            'Orbit Eccentricity',
            'Orbit Inclination, °',
            'Ex. Long. Asc. Node, °',
            'Ex. Arg. of Periapsis, °',
            'Exoplanet Radius, km',
            'Ring Eccentricity',
            'Ring Semi-Major Axis, km',
            'Ring Width, km',
            'Ring Obliquity, °',
            'Ring Azimuthal Angle, °',
            'Ring Arg. of Periapsis, °'
        ]
        
        print(f"Loaded {len(self.samples)} samples with {len(self.labels)} parameters")
        print(f"Log-evidence: {self.logz[-1]:.2f}")
        
    def load_observations(self):
        """Load observational data."""
        files = {
            "C18 short cadence": ("observations/C18_short_cadence.csv", "red"),
            "C18 long cadence": ("observations/C18_long_cadence.csv", "green"),
            "C5": ("observations/C5.csv", "blue"),
        }
        
        all_times = []
        all_mag_changes = []
        
        # Load data from each file
        for label, (filename, color) in files.items():
            try:
                df = pd.read_csv(filename, header=None, names=['time', 'flux'])
                mag_change = -2.5 * np.log10(df['flux'])
                all_times.extend(df['time'].tolist())
                all_mag_changes.extend(mag_change.tolist())
            except FileNotFoundError:
                print(f"Warning: Could not find the file {filename}. Skipping.")
                continue
            except Exception as e:
                print(f"An error occurred while reading {filename}: {e}")
                continue
        
        if not all_times:
            print("No data was loaded. Exiting.")
            sys.exit(1)
        
        all_times = np.array(all_times)
        all_mag_changes = np.array(all_mag_changes)
        observations_duration = np.max(all_times) - np.min(all_times)
        
        # Calculate phase, wrapping around 0 to 1
        phases = (all_times / observations_duration) + 0.5
        
        # Return combined data, sorted by phase
        sorted_indices = np.argsort(phases)
        self.observations = np.vstack((phases[sorted_indices], all_mag_changes[sorted_indices])).T

    def kfold_split(self, n_splits: int, removed: List[int], _dir: str = 'kfold_observations.jpg',
                    title: str = 'K-Fold Observational Light Curve',
                    x_label: str = 'Phase',
                    y_label: str = 'Magnitude Change',
                    invert_x: bool = False,
                    invert_y: bool = True):
        """
        Loads observational data from specified CSV files, performs k-fold splitting,
        removes specified chunks, converts flux to magnitude change, calculates the orbital phase,
        and generates a plot of the remaining data.

        Args:
            n_splits: Number of chunks to split the data into
            removed: List of chunk indices to remove (0-based indexing)
            _dir: Output filename for the plot
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            invert_x: Whether to invert the x-axis
            invert_y: Whether to invert the y-axis
        """
        print(f"Loading and processing observational data with {n_splits}-fold splitting...")
        print(f"Removing chunks: {removed}")

        all_times = []
        all_mag_changes = []

        plt.figure(figsize=(12, 7))

        files = {
            "C18 short cadence": ("observations/C18_short_cadence.csv", "red"),
            "C18 long cadence": ("observations/C18_long_cadence.csv", "green"),
            "C5": ("observations/C5.csv", "blue"),
        }

        # Load data from each file
        for label, (filename, color) in files.items():
            try:
                df = pd.read_csv(filename, header=None, names=['time', 'flux'])
                mag_change = -2.5 * np.log10(df['flux'])

                all_times.extend(df['time'].tolist())
                all_mag_changes.extend(mag_change.tolist())

            except FileNotFoundError:
                print(f"Warning: Could not find the file {filename}. Skipping.")
                continue
            except Exception as e:
                print(f"An error occurred while reading {filename}: {e}")
                continue

        if not all_times:
            print("No data was loaded. Exiting.")
            sys.exit(1)

        all_times = np.array(all_times)
        all_mag_changes = np.array(all_mag_changes)
        observations_duration = np.max(all_times) - np.min(all_times)

        phases = (all_times / observations_duration) + 0.5

        sorted_indices = np.argsort(phases)
        sorted_phases = phases[sorted_indices]
        sorted_mag_changes = all_mag_changes[sorted_indices]

        chunk_size = len(sorted_phases) // n_splits
        chunks_phases = []
        chunks_mag_changes = []

        for i in range(n_splits):
            start_idx = i * chunk_size
            if i == n_splits - 1:  # Last chunk gets remaining data
                end_idx = len(sorted_phases)
            else:
                end_idx = (i + 1) * chunk_size

            chunks_phases.append(sorted_phases[start_idx:end_idx])
            chunks_mag_changes.append(sorted_mag_changes[start_idx:end_idx])

        remaining_phases = []
        remaining_mag_changes = []
        for i, (chunk_phases, chunk_mag_changes) in enumerate(zip(chunks_phases, chunks_mag_changes)):
            if i not in removed:
                remaining_phases.extend(chunk_phases)
                remaining_mag_changes.extend(chunk_mag_changes)

        if not remaining_phases:
            print("All chunks were removed. No data remaining.")
            sys.exit(1)

        remaining_phases = np.array(remaining_phases)
        remaining_mag_changes = np.array(remaining_mag_changes)

        # Plot the remaining data
        plt.plot(remaining_phases, remaining_mag_changes, 'o',
                 markersize=3, color='blue', alpha=0.7, label='Training Data')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{title} (Chunks {list(set(range(n_splits)) - set(removed))} retained)")
        plt.grid(True)
        plt.legend()
        if invert_x:
            plt.gca().invert_xaxis()
        if invert_y:
            plt.gca().invert_yaxis()
        plt.savefig(_dir)
        plt.close()
        print(f"K-fold observational data plot saved to {_dir}")

        sorted_remaining_indices = np.argsort(remaining_phases)
        self.observations = np.vstack((remaining_phases[sorted_remaining_indices],
                                       remaining_mag_changes[sorted_remaining_indices])).T
        
    def get_posterior_samples(self, nsamples=1000):
        """Get posterior samples from the nested sampling results."""
        # Get equally weighted posterior samples
        weights = np.exp(self.logwt - self.logz[-1])
        samples = dyfunc.resample_equal(
            samples=self.samples,
            weights=weights
        )
        return samples[:nsamples]
    
    def summary(self, q1=16, q2=50, q3=84):
        """Generate summary statistics for the parameters."""
        samples = self.get_posterior_samples()
        ans = ''
        for i in range(len(self.labels)):
            percentiles = np.percentile(samples[:, i], [q1, q2, q3])
            
            # Calculate posterior standard deviation (approximate from percentiles)
            # Using the interquartile range divided by 1.349 as a robust estimator
            posterior_stdev = (percentiles[2] - percentiles[0]) / 1.349
            
            # Calculate shrinkage:
            shrinkage = 1 - (posterior_stdev / self.prior_stdev[i])
            
            ans += f"{self.labels[i]:<28}: {percentiles[1]:.3f} +{percentiles[2] - percentiles[1]:.3f} / -{percentiles[1] - percentiles[0]:.3f} (shrinkage: {shrinkage:.4f})\n"
        
        ans += f"\nLog-evidence: {self.logz[-1]:.2f} +/- {np.std(self.logz):.2f}\n"
        
        return ans

    def corner_plot(self, filename: str = 'corner_plot.png', n_prior_samples: int = 1000):
        """
        Create a corner plot showing both prior and posterior distributions.
        """
        print("\nGenerating corner plot...")

        try:
            # Get posterior samples
            post_samples = self.get_posterior_samples()

            if post_samples is None or len(post_samples) == 0:
                print("No posterior samples available for corner plot.")
                return

            ndim = post_samples.shape[1]
            # Access the boundaries defined in the global specific_parameter_boundaries
            param_bounds = list(specific_parameter_boundaries.values())

            # Calculate ranges using min/max with padding to focus on the posterior
            ranges = []
            for i in range(ndim):
                min_val, max_val = np.min(post_samples[:, i]), np.max(post_samples[:, i])
                padding = 0.1 * (max_val - min_val) if max_val != min_val else 0.1
                ranges.append((min_val - padding, max_val + padding))

            # Create the corner plot
            fig = corner.corner(
                post_samples,
                labels=self.labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt=".3f",
                title_kwargs={"fontsize": 10},
                label_kwargs={"fontsize": 12},
                labelpad=0.25,  # Increased padding to move labels down
                plot_datapoints=True,
                plot_density=True,
                plot_contours=True,
                fill_contours=True,
                levels=1.0 - np.exp(-0.5 * np.array([1.0, 2.0]) ** 2),
                alpha=0.5,
                bins=30,
                smooth=1.0,
                smooth1d=1.0,
                range=ranges
            )

            # Add uniform prior lines to the 1D histograms (the diagonal)
            axes = np.array(fig.axes).reshape((ndim, ndim))
            for i in range(ndim):
                ax = axes[i, i]
                low, high = param_bounds[i]

                # The height of a normalized uniform distribution is 1 / range
                prior_height = 1.0 / (high - low)

                # Plot the prior as a thick blue line
                # Note: Only the portion within the 'range' of the axis will be visible
                ax.plot([low, high], [prior_height, prior_height],
                        color='blue', linestyle='-', linewidth=2.5,
                        label='Uniform Prior', alpha=0.7)

            # Add a global title
            fig.suptitle('Corner Plot for Posterior Distributions vs. Uniform Priors', y=1.02, fontsize=16)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

            # Save the figure with tight bounding box to prevent label truncation
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Corner plot with prior overlays saved to {os.path.abspath(filename)}")

        except Exception as e:
            print(f"Error generating corner plot: {str(e)}")
            raise
    
    def sample_prior(self, nsamples=1000):
        """Sample from the prior distribution."""
        param_bounds = list(specific_parameter_boundaries.values())
        # Sample from unit cube
        u_samples = np.random.uniform(0, 1, size=(nsamples, len(param_bounds)))
        # Transform to parameter space
        return np.array([self.prior_transform(u, param_bounds) for u in u_samples])
    
    def prior_transform(self, u, param_bounds):
        """Transform from unit cube to physical parameter space."""
        params = np.zeros_like(u)
        for i, (u_i, (low, high)) in enumerate(zip(u, param_bounds)):
            params[i] = low + (high - low) * u_i
        return params
    
    def best_fit_vs_observations(self, filename: str = 'best_fit_model_vs_observed_data.png'):
        """Plot the best-fit model against the observed data."""
        print("\nGenerating best-fit model vs. observed data plot...")
        
        # Get the sample with the highest likelihood
        best_fit_params = self.samples[np.argmax(self.logl)]
        
        best_fit_model_output, transit_duration, _ = calculate_data(
            fixed_exoplanet_sma,
            best_fit_params[0],  # exoplanet_orbit_eccentricity
            best_fit_params[1],  # exoplanet_orbit_inclination
            best_fit_params[2],  # exoplanet_longitude_of_ascending_node
            best_fit_params[3],  # exoplanet_argument_of_periapsis
            best_fit_params[4],  # exoplanet_radius
            fixed_exoplanet_mass,
            roche_density(fixed_exoplanet_mass, best_fit_params[6], best_fit_params[5]),  # density
            best_fit_params[5],  # eccentricity
            best_fit_params[6],  # sma
            best_fit_params[7],  # width
            np.nan,
            best_fit_params[8],  # obliquity
            best_fit_params[9],  # azimuthal_angle
            best_fit_params[10],  # argument_of_periapsis
            fixed_specific_absorption_coefficient,
            fixed_star_object,
            fixed_pixel_size,
            custom_units=False
        )
        
        best_fit_model_lightcurve = np.array(best_fit_model_output)
        
        # Process observations
        observed_phases = (self.observations[:, 0] - 0.5) * 0.5 * fixed_observations_duration / transit_duration + 0.5
        observed_magnitudes = self.observations[:, 1]
        min_phase = np.maximum(np.min(observed_phases), np.min(best_fit_model_lightcurve[:, 0]))
        max_phase = np.minimum(np.max(observed_phases), np.max(best_fit_model_lightcurve[:, 0]))
        intersection = (observed_phases >= min_phase) & (observed_phases <= max_phase)
        obs_phase_intersect = observed_phases[intersection]
        obs_mag_intersect = observed_magnitudes[intersection]
        
        # Calculate residuals
        model_interp = np.interp(
            obs_phase_intersect,
            best_fit_model_lightcurve[:, 0],
            best_fit_model_lightcurve[:, 1]
        )
        residuals = obs_mag_intersect - model_interp
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                       gridspec_kw={'height_ratios': [2, 1]},
                                       sharex=True)
        
        # Top panel: Data and model
        ax1.plot(obs_phase_intersect, obs_mag_intersect, 'o',
                 color='blue', markersize=3, alpha=0.5, label='Observed Data')
        ax1.plot(best_fit_model_lightcurve[:, 0], best_fit_model_lightcurve[:, 1],
                 '-', color='black', linewidth=2, label='Best-Fit Model')
        ax1.set_ylabel('Magnitude Change')
        ax1.legend()
        ax1.set_title('Best-Fit Model vs. Observed Data')
        ax1.grid(True)
        ax1.invert_yaxis()
        
        # Bottom panel: Residuals
        ax2.plot(obs_phase_intersect, residuals, 'o',
                 color='red', markersize=3, alpha=0.5)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Residuals')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Best-fit model vs. observed data plot saved to {filename}")
    
    def ppc(self, n_samples=100, filename='ppc.png', deviation=0.02):
        """
        Performs and plots posterior-predictive check for the best-fit data.
        """
        print("\nGenerating posterior predictive check...")
        
        best_fit_params = self.samples[np.argmax(self.logl)]
        
        fig, ax = plt.subplots()
        ax.set_ylabel('Magnitude Change')
        ax.set_title('Posterior Predictive Check')
        plt.grid(True)
        ax.invert_yaxis()
        
        best_fit_model_output, transit_duration, _ = calculate_data(
            fixed_exoplanet_sma,
            best_fit_params[0],  # exoplanet_orbit_eccentricity
            best_fit_params[1],  # exoplanet_orbit_inclination
            best_fit_params[2],  # exoplanet_longitude_of_ascending_node
            best_fit_params[3],  # exoplanet_argument_of_periapsis
            best_fit_params[4],  # exoplanet_radius
            fixed_exoplanet_mass,
            roche_density(fixed_exoplanet_mass, best_fit_params[6], best_fit_params[5]),  # density
            best_fit_params[5],  # eccentricity
            best_fit_params[6],  # sma
            best_fit_params[7],  # width
            np.nan,
            best_fit_params[8],  # obliquity
            best_fit_params[9],  # azimuthal_angle
            best_fit_params[10],  # argument_of_periapsis
            fixed_specific_absorption_coefficient,
            fixed_star_object,
            fixed_pixel_size,
            custom_units=False
        )
        
        best_fit_model_lightcurve = np.array(best_fit_model_output)
        predictives = []
        
        for _ in range(n_samples):
            sample = best_fit_params.copy()
            for i in range(len(sample)):
                sample[i] = random.uniform(sample[i] * (1 - deviation), sample[i] * (1 + deviation))
            best_fit_model_output, transit_duration, _ = calculate_data(
                fixed_exoplanet_sma,
                sample[0],  # exoplanet_orbit_eccentricity
                sample[1],  # exoplanet_orbit_inclination
                90,  # exoplanet_longitude_of_ascending_node
                sample[3],  # exoplanet_argument_of_periapsis
                sample[4],  # exoplanet_radius
                fixed_exoplanet_mass,
                roche_density(fixed_exoplanet_mass, best_fit_params[6], best_fit_params[5]),  # density
                sample[5],  # eccentricity
                sample[6],  # sma
                sample[7],  # width
                np.nan,
                sample[8],  # obliquity
                sample[9],  # azimuthal_angle
                sample[10],  # argument_of_periapsis
                fixed_specific_absorption_coefficient,
                fixed_star_object,
                fixed_pixel_size,
                custom_units=False
            )
            
            model_lightcurve = np.array(best_fit_model_output)
            predictives.append(model_lightcurve[:, 1])
            
            plt.plot(model_lightcurve[:, 0], model_lightcurve[:, 1],
                     '-', color='blue', alpha=0.1, label='Posterior Predictive')
        
        plt.plot(best_fit_model_lightcurve[:, 0], np.mean(np.vstack(predictives), axis=0), '--', color='orange', label="Posterior Predictive Mean")
        plt.plot(best_fit_model_lightcurve[:, 0], best_fit_model_lightcurve[:, 1],
                 '-', color='black', linewidth=2, label='Best-Fit Model')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(filename)
        plt.close()
        print(f"Posterior predictive check plot saved to {filename}")
    
    def trace_plots(self, filename='trace_plots.png'):
        """Generate trace plots for the MCMC chains."""
        print("\nGenerating trace plots...")
        
        samples = self.get_posterior_samples(500)
        
        fig, axes = plt.subplots(len(self.labels), 1, figsize=(10, 2*len(self.labels)))
        if len(self.labels) == 1:
            axes = [axes]
        
        for i, (ax, label) in enumerate(zip(axes, self.labels)):
            ax.plot(samples[:, i], 'b-', alpha=0.7)
            ax.set_ylabel(label)
            ax.grid(True)
        
        ax.set_xlabel('Sample Number')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Trace plots saved to {filename}")
    
    def parameter_distributions(self, filename='parameter_distributions.png'):
        """Plot parameter distributions with statistics."""
        print("\nGenerating parameter distribution plots...")
        
        samples = self.get_posterior_samples()
        
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (ax, label) in enumerate(zip(axes[:len(self.labels)], self.labels)):
            ax.hist(samples[:, i], bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.grid(True)
            
            # Add statistics
            mean = np.mean(samples[:, i])
            std = np.std(samples[:, i])
            median = np.median(samples[:, i])
            
            ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
            ax.axvline(median, color='green', linestyle='--', label=f'Median: {median:.3f}')
            ax.legend()
        
        # Hide unused subplots
        for i in range(len(self.labels), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Parameter distributions saved to {filename}")
    
    def correlation_heatmap(self, filename='correlation_heatmap.png'):
        """Generate a correlation heatmap of parameters."""
        print("\nGenerating correlation heatmap...")
        
        samples = self.get_posterior_samples()
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(samples.T)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                   xticklabels=self.labels,
                   yticklabels=self.labels,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   vmin=-1, 
                   vmax=1)
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Correlation heatmap saved to {filename}")
    
    def analyze(self, folder: str = ''):
        """Run all analyses and save results."""
        print("Running complete nested sampling analysis...")
        
        # Set up plotting style
        sns.set_theme(style="whitegrid")
        
        # Run all analyses
        self.corner_plot(folder + 'corner_plot.png')
        self.best_fit_vs_observations(folder + 'best_fit_model_vs_observed_data.png')
        self.ppc(filename=folder + 'ppc.png')
        self.trace_plots(folder + 'trace_plots.png')
        self.parameter_distributions(folder + 'parameter_distributions.png')
        self.correlation_heatmap(folder + 'correlation_heatmap.png')
        
        # Generate summary statistics
        stats = self.summary()
        print("\nParameter estimates (50th percentile) with 1-sigma uncertainties:")
        print(stats)
        
        # Save summary to file
        with open(folder + 'nested_sampling_analysis_summary.txt', 'w') as f:
            f.write("Complete Nested Sampling Analysis Results\n")
            f.write("==========================================\n\n")
            f.write("Parameter estimates (50th percentile) with 1-sigma uncertainties:\n")
            f.write(stats)
            
            # Add best-fit parameters
            best_fit_params = self.samples[np.argmax(self.logl)]
            f.write("\n\nBest-fit Parameters:\n")
            for i, label in enumerate(self.labels):
                f.write(f"{label}: {best_fit_params[i]:.6f}\n")
            
            f.write(f"\nMaximum Log-Likelihood: {np.max(self.logl):.2f}\n")
            f.write(f"Log-Evidence: {self.logz[-1]:.2f}\n")
        
        print("Analysis complete! All plots and summary saved.")
        return stats

if __name__ == "__main__":
    nsa = NestedSamplingAnalysis('nested_sampling_run/nested_sampling_results.npz')
    nsa.analyze('nested_sampling_run/')
