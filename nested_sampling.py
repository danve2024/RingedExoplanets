import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import corner
import sys
from typing import Union, List, Dict
from dynesty import DynamicNestedSampler
from dynesty import utils as dyfunc
from data_fitting import fixed_exoplanet_mass
from formulas import roche_sma_max, roche_density, volume
from main import calculate_data, defaults
from models import quadratic_star_model
from units import *
from space import CustomStarModel
import os
import json
import random
from collections import OrderedDict

invalid = False

max_semi_major_axis_max = roche_sma_max(50_000 * km, fixed_exoplanet_mass/volume(50_000 * km), 0.3, roche_density(fixed_exoplanet_mass, 40_000 * 0.9 * km, 0.4))
max_width_max = max_semi_major_axis_max - 40_000 * 0.9 * km

specific_parameter_boundaries = {
    'exoplanet_orbit_eccentricity': (0, 0.1),
    'exoplanet_orbit_inclination': (85, 90),
    'exoplanet_longitude_of_ascending_node': (90 - 5e-8, 90 + 5e-8),
    'exoplanet_argument_of_periapsis': (0, 0.1),
    'exoplanet_radius': (40_000 * km, 50_000 * km),
    'eccentricity': (0.3, 0.4),
    'semi_major_axis': (40_000 * 0.9 * km, max_semi_major_axis_max), # dynamic boundaries
    'width': (0, max_width_max), # dynamic boundaries
    'obliquity': (85, 90),
    'azimuthal_angle': (0, 0.1),
    'argument_of_periapsis': (0, 0.1)
}


class NestedSampler:
    def __init__(self, nlive: int, ndim: int, files: Dict, labels: List[str], loglike_file: str = 'nested_sampling/loglike.json', best_fit_params: Union[list, np.ndarray] = None):
        """
        Initialize the nested sampler.

        Args:
            nlive: Number of live points
            ndim: Number of dimensions
            files: Dictionary of observation files and their colors
            labels: List of parameter names
            best_fit_params: Best-fit parameters for result analyses (required only if the sampler is not run)
        """
        print("Initializing Nested Sampler...")
        self.nlive = nlive
        self.ndim = ndim
        self.files = files
        self.labels = labels
        self.observations = None
        self.load_and_plot_observations()
        self.results = None
        self.loglike_file = loglike_file
        self.best_fit_params = best_fit_params

        with open(self.loglike_file, 'r') as file:
            self.loglikes = json.load(file)

        # Store parameter bounds for prior transform
        self.param_bounds = list(specific_parameter_boundaries.values())

        # Initialize the dynamic nested sampler
        self.sampler = DynamicNestedSampler(
            self.log_likelihood,
            self.prior_transform,
            ndim=ndim,
            nlive=nlive
        )

    def prior_transform(self, u):
        """Transform from unit cube to physical parameter space."""
        params = np.zeros_like(u)
        for i, (u_i, (low, high)) in enumerate(zip(u, self.param_bounds)):
            params[i] = low + (high - low) * u_i
        return params

    def log_likelihood(self, params):
        """
        Calculates the log-likelihood of the observed_data given the model parameters.

        Returns:
            float: The log-likelihood value. Returns -np.inf if calculation fails or is invalid.
        """

        key = ' '.join(list(map(str, params)))
        log_likelihood_value = self.loglikes.get(key)

        if log_likelihood_value is None:

            exoplanet_orbit_eccentricity, exoplanet_orbit_inclination, \
                exoplanet_longitude_of_ascending_node, exoplanet_argument_of_periapsis, \
                exoplanet_radius, eccentricity, sma, width, obliquity, azimuthal_angle, \
                argument_of_periapsis = params

            density = roche_density(fixed_exoplanet_mass, sma, eccentricity)
            exoplanet_volume = volume(exoplanet_radius)
            exoplanet_density = fixed_exoplanet_mass / exoplanet_volume

            a_max = roche_sma_max(exoplanet_radius, exoplanet_density, eccentricity, density)
            if sma >= a_max:
                print(f'Warning: parameter semi-major axis out of boundaries: {sma}.')
                return -np.inf

            w_max = a_max - sma
            if width >= w_max:
                print(f'Warning: parameter width out of boundaries: {width}.')
                return -np.inf

            density = roche_density(fixed_exoplanet_mass, sma, eccentricity)

            model_output, transit_duration, exoplanet_obj = calculate_data(fixed_exoplanet_sma,
                                                                           exoplanet_orbit_eccentricity,
                                                                           exoplanet_orbit_inclination,
                                                                           exoplanet_longitude_of_ascending_node,
                                                                           exoplanet_argument_of_periapsis,
                                                                           exoplanet_radius,
                                                                           fixed_exoplanet_mass,
                                                                           density,
                                                                           eccentricity,
                                                                           sma,
                                                                           width,
                                                                           np.nan,
                                                                           obliquity,
                                                                           azimuthal_angle,
                                                                           argument_of_periapsis,
                                                                           fixed_specific_absorption_coefficient,
                                                                           fixed_star_object,
                                                                           fixed_pixel_size,
                                                                           custom_units=False
                                                                           )
            model_lightcurve = np.array(model_output)

            if model_lightcurve.shape[0] < 2:
                return -np.inf

            observed_phases = (self.observations[:, 0] - 0.5) * 0.5 * fixed_observations_duration / transit_duration + 0.5
            observed_magnitudes = self.observations[:, 1]
            min_phase = np.maximum(np.min(observed_phases), np.min(model_lightcurve[:, 0]))
            max_phase = np.minimum(np.max(observed_phases), np.max(model_lightcurve[:, 0]))
            intersection = (observed_phases >= min_phase) & (observed_phases <= max_phase)
            obs_phase_intersect = observed_phases[intersection]
            obs_mag_intersect = observed_magnitudes[intersection]

            if len(obs_phase_intersect) == 0: return -np.inf

            model_predicted_magnitudes = np.interp(
                obs_phase_intersect,
                model_lightcurve[:, 0],
                model_lightcurve[:, 1]
            )

            residuals = obs_mag_intersect - model_predicted_magnitudes
            out_of_transit_mask = (obs_mag_intersect > np.percentile(obs_mag_intersect, 90))
            sigma2 = np.var(obs_mag_intersect[out_of_transit_mask])

            if sigma2 == 0:
                self.loglikes[key] = -np.inf
                return -np.inf

            log_likelihood_value = -0.5 * np.sum((residuals ** 2) / sigma2 + np.log(2 * np.pi * sigma2))

            if not np.isfinite(log_likelihood_value):
                self.loglikes[key] = -np.inf
                return -np.inf

            self.loglikes[key] = log_likelihood_value

        print(log_likelihood_value)
        return log_likelihood_value

    def best_log_likelihood(self):
        return self.log_likelihood(self.best_fit_params)

    def run(self):
        """Run the nested sampling."""
        print(f"Running dynamic nested sampling with {self.nlive} live points...")
        self.sampler.run_nested()
        self.results = self.sampler.results
        print("Nested sampling completed.")
        self.best_fit_params = self.results.samples[np.argmax(self.results.logl)]
        return self.results

    def save(self, filename: str = "nested_sampling_results.npz"):
        with open(self.loglike_file, 'w') as file:
            json.dump(self.loglikes, file)
        np.savez_compressed(filename,
                            samples=self.results.samples,
                            logwt=self.results.logwt,
                            logz=self.results.logz,
                            logl=self.results.logl,
                            labels=self.labels)

    def get_posterior_samples(self, nsamples=1000):
        """Get posterior samples from the nested sampling results."""
        # Get equally weighted posterior samples
        weights = np.exp(self.results.logwt - self.results.logz[-1])
        samples = dyfunc.resample_equal(
            samples=self.results.samples,
            weights=weights
        )
        return samples[:nsamples]

    def summary(self, q1=16, q2=50, q3=84):
        """Generate summary statistics for the parameters."""
        samples = self.get_posterior_samples()
        ans = ''
        for i in range(self.ndim):
            percentiles = np.percentile(samples[:, i], [q1, q2, q3])
            ans += f"{self.labels[i]:<28}: {percentiles[1]:.3f} +{percentiles[2] - percentiles[1]:.3f} / -{percentiles[1] - percentiles[0]:.3f}\n"

        ans += f"\nLog-evidence: {results.logz[-1]:.2f} +/- {results.logzerr[-1]:.2f}\n"

        return ans

    def sample_prior(self, nsamples=1000):
        """Sample from the prior distribution."""
        # Sample from unit cube
        u_samples = np.random.uniform(0, 1, size=(nsamples, self.ndim))
        # Transform to parameter space
        return np.array([self.prior_transform(u) for u in u_samples])

    def corner_plot(self, filename: str = 'corner_plot.png', n_prior_samples: int = 1000):
        """
        Create a corner plot showing both prior and posterior distributions.

        Args:
            filename: Output filename for the corner plot
            n_prior_samples: Number of prior samples to generate for visualization
        """
        if self.results is None:
            print("No results available. Run the sampler first.")
            return

        print("\nGenerating corner plot with priors...")

        try:
            # Get posterior samples
            post_samples = self.get_posterior_samples()

            if post_samples is None or len(post_samples) == 0:
                print("No posterior samples available for corner plot.")
                return

            # Generate prior samples
            prior_samples = self.sample_prior(n_prior_samples)

            if prior_samples is None or len(prior_samples) == 0:
                print("Warning: Could not generate prior samples. Plotting posterior only.")
                prior_samples = None

            # Calculate ranges using min/max with padding
            ranges = []
            for i in range(post_samples.shape[1]):
                min_val, max_val = np.min(post_samples[:, i]), np.max(post_samples[:, i])
                padding = 0.1 * (max_val - min_val) if max_val != min_val else 0.1
                ranges.append((min_val - padding, max_val + padding))

            # Create the base corner plot with posterior
            fig = corner.corner(
                post_samples,
                labels=self.labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt=".3f",
                title_kwargs={"fontsize": 10},
                label_kwargs={"fontsize": 12},
                plot_datapoints=True,
                plot_density=True,
                plot_contours=True,
                fill_contours=True,
                levels=1.0 - np.exp(-0.5 * np.array([1.0, 2.0]) ** 2),
                color='#0072B2',  # Blue for posterior
                alpha=0.5,
                bins=30,
                smooth=1.0,
                smooth1d=1.0,
                hist_kwargs={'density': True, 'color': '#4D8DC4', 'histtype': 'stepfilled'},
                range=ranges
            )

            # Add prior distributions if available
            if prior_samples is not None:
                corner.corner(
                    prior_samples,
                    fig=fig,
                    color='#D55E00',  # Red for prior
                    alpha=0.3,
                    bins=20,
                    smooth=1.0,
                    smooth1d=1.0,
                    hist_kwargs={'density': True, 'color': '#E69F00', 'alpha': 0.5, 'histtype': 'stepfilled'},
                    plot_density=False,
                    plot_datapoints=False,
                    plot_contours=False,
                    no_fill_contours=True
                )

                # Add legend
                plt.figure(fig.number)
                plt.figtext(0.7, 0.9, 'Posterior', color='#0072B2', fontsize=12, ha='center')
                plt.figtext(0.8, 0.9, 'Prior', color='#D55E00', fontsize=12, ha='center')

            # Adjust layout
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.1, hspace=0.1)

            # Add title
            fig.suptitle('Prior vs Posterior Parameter Distributions', y=0.98, fontsize=16)

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

            # Save the figure
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Corner plot with priors saved to {os.path.abspath(filename)}")

        except Exception as e:
            print(f"Error generating corner plot: {str(e)}")
            if 'post_samples' in locals():
                print(f"Posterior samples shape: {post_samples.shape if post_samples is not None else 'None'}")
            if 'prior_samples' in locals():
                print(
                    f"Prior samples shape: {prior_samples.shape if 'prior_samples' in locals() and prior_samples is not None else 'None'}")
            print(f"Labels: {self.labels}")
            raise
        print(f"Corner plot with priors saved to {filename}")

    def best_fit_vs_observations(self, filename: str = 'best_fit_model_vs_observed_data.png'):
        """Plot the best-fit model against the observed data."""
        print("\nGenerating best-fit model vs. observed data plot...")

        best_fit_model_output, transit_duration, _ = calculate_data(
            fixed_exoplanet_sma,
            self.best_fit_params[0],  # exoplanet_orbit_eccentricity
            self.best_fit_params[1],  # exoplanet_orbit_inclination
            self.best_fit_params[2],  # exoplanet_longitude_of_ascending_node
            self.best_fit_params[3],  # exoplanet_argument_of_periapsis
            self.best_fit_params[4],  # exoplanet_radius
            fixed_exoplanet_mass,
            roche_density(fixed_exoplanet_mass, self.best_fit_params[6], self.best_fit_params[5]),  # density
            self.best_fit_params[5],  # eccentricity
            self.best_fit_params[6],  # sma
            self.best_fit_params[7],  # width
            np.nan,
            self.best_fit_params[8],  # obliquity
            self.best_fit_params[9],  # azimuthal_angle
            self.best_fit_params[10],  # argument_of_periapsis
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

    def load_and_plot_observations(self, _dir: str = 'observations.jpg',
                                   title: str = 'Combined Observational Light Curve',
                                   x_label: str = 'Phase',
                                   y_label: str = 'Magnitude Change',
                                   invert_x: bool = False,
                                   invert_y: bool = True):
        """
        Loads observational data from specified CSV files, converts flux to magnitude change,
        calculates the orbital phase, and generates a plot of the combined data.
        """
        print("Loading and processing observational data from CSV files...")

        all_times = []
        all_mag_changes = []

        plt.figure(figsize=(12, 7))

        # Load data from each file
        for label, (filename, color) in self.files.items():
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

        # Plot each dataset by phase
        start_index = 0
        for label, (filename, color) in self.files.items():
            try:
                df = pd.read_csv(filename, header=None)
                num_points = len(df)
                end_index = start_index + num_points

                # Get the phases corresponding to this dataset
                dataset_phases = phases[start_index:end_index]
                dataset_mag_changes = all_mag_changes[start_index:end_index]

                plt.plot(dataset_phases, dataset_mag_changes, 'o',
                         markersize=3, color=color, label=label, alpha=0.7)
                start_index = end_index
            except FileNotFoundError:
                continue

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        if invert_x:
            plt.gca().invert_xaxis()
        if invert_y:
            plt.gca().invert_yaxis()
        plt.savefig(_dir)
        plt.close()
        print(f"Observational data plot saved to {_dir}")

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

        # Load data from each file
        for label, (filename, color) in self.files.items():
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

    def ppc(self, n_samples=100, filename='ppc.png', deviation=0.02):
        """
        Performs and plots posterior-predictive check for the best-fit data.

        :param n_samples: the total number of samples for posterior-predictive check
        :param filename: the directory to save the graph
        :param deviation: maximum possible deviation of the parameters in the samples from the best-fit
        :return:
        """

        fig, ax = plt.subplots()
        ax.set_ylabel('Magnitude Change')
        ax.set_title('Posterior Predictive Check')
        plt.grid(True)
        ax.invert_yaxis()

        best_fit_model_output, transit_duration, _ = calculate_data(
            fixed_exoplanet_sma,
            self.best_fit_params[0],  # exoplanet_orbit_eccentricity
            self.best_fit_params[1],  # exoplanet_orbit_inclination
            self.best_fit_params[2],  # exoplanet_longitude_of_ascending_node
            self.best_fit_params[3],  # exoplanet_argument_of_periapsis
            self.best_fit_params[4],  # exoplanet_radius
            fixed_exoplanet_mass,
            roche_density(fixed_exoplanet_mass, self.best_fit_params[6], self.best_fit_params[5]),  # density
            self.best_fit_params[5],  # eccentricity
            self.best_fit_params[6],  # sma
            self.best_fit_params[7],  # width
            np.nan,
            self.best_fit_params[8],  # obliquity
            self.best_fit_params[9],  # azimuthal_angle
            self.best_fit_params[10],  # argument_of_periapsis
            fixed_specific_absorption_coefficient,
            fixed_star_object,
            fixed_pixel_size,
            custom_units=False
        )

        best_fit_model_lightcurve = np.array(best_fit_model_output)
        predictives = []

        for _ in range(n_samples):
            sample = self.best_fit_params.copy()
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
                roche_density(fixed_exoplanet_mass, self.best_fit_params[6], self.best_fit_params[5]),  # density
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

    def analyze(self):
        self.corner_plot()
        self.ppc()

        # Print summary statistics
        stats = self.summary()
        print("Parameter estimates (50th percentile) with 1-sigma uncertainties:")
        print(stats)

        with open('nested_sampling_summary.txt', 'w') as f:
            f.write("Nested Sampling Results\n")
            f.write("======================\n\n")
            f.write("Parameter estimates (50th percentile) with 1-sigma uncertainties:\n")
            f.write(stats)

        print("Analysis complete!")



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

sns.set_theme(style="whitegrid")

nlive = 500
ndim = 11

# Initialize and run the nested sampler
ns = NestedSampler(
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
        'Ring Eccentricity',
        'Ring Obliquity, °',
        'Ring Azimuthal Angle, °',
        'Ring Arg. of Periapsis, °'
    ]
)

if __name__ == "__main__":
    results = ns.run()
    ns.save()
    ns.analyze()
