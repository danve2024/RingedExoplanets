import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emcee
import corner
import sys
from typing import Union
from formulas import roche_sma_max, roche_density, volume
from main import calculate_data, defaults
from models import quadratic_star_model
from units import *
from space import CustomStarModel

invalid = False

specific_parameter_boundaries = {
    'exoplanet_orbit_eccentricity': (0, 0.1),
    'exoplanet_orbit_inclination': (80, 90),
    'exoplanet_longitude_of_ascending_node': (90 - 5e-8, 90 + 5e-8),
    'exoplanet_argument_of_periapsis': (0, 360),
    'exoplanet_radius': (8000 * km, 60000 * km),
    'eccentricity': (0.0, 0.4),
    'obliquity': (0, 90),
    'azimuthal_angle': (0, 360),
    'argument_of_periapsis': (0, 360)
}

class MCMC(emcee.EnsembleSampler):
    def __init__(self, nwalkers: int, ndim: int, iterations: int, burn_in: int, thinning: int, files: dict, labels: list[str], initial_guesses: Union[list, np.ndarray]):
        print("Initializing MCMC sampler...")
        self.nwalkers = nwalkers,
        self.ndim = ndim
        self.iterations = iterations
        self.burn_in = burn_in
        self.thinning = thinning
        self.files = files
        self.observations = None
        self.load_and_plot_observations()
        self.initial_parameters = initial_guesses + 1e-8 * np.random.randn(nwalkers, ndim)
        self.labels = labels
        self.display_units = {
            label: km for label in self.labels if 'km' in label
        }
        super().__init__(nwalkers, ndim, self.log_posterior)

    def run(self):
        print(f"Running MCMC simulation with {self.nwalkers} walkers for {self.iterations} steps...")
        self.run_mcmc(self.initial_parameters, self.iterations, progress=True)
        print("MCMC simulation finished.")

    def save_chain(self, filename='mcmc_chain_results.npy'):
        np.save(filename, self.get_chain())

    def get_display_samples(self, flat=True, raw_chain=False):
        """
        Returns the MCMC samples with units converted for display purposes.
        This does NOT affect the actual calculations.

        Args:
            flat (bool): If True, returns flattened (1D) effective samples.
            raw_chain (bool): If True, returns the full, unconverged chain.

        Returns:
            np.ndarray: The samples with display unit conversions applied.
        """
        if raw_chain:
            samples = self.get_chain()
        else:
            samples = self.get_chain(discard=self.burn_in, thin=self.thinning, flat=flat)

        converted_samples = np.copy(samples)

        for i, label in enumerate(self.labels):
            if label in self.display_units:
                factor = self.display_units[label]
                if raw_chain:
                    converted_samples[:, :, i] /= factor
                else:
                    converted_samples[..., i] /= factor
        return converted_samples

    def get_effective_samples(self):
        print(f"\nApplying burn-in of {self.burn_in} iterations and thinning by {self.thinning}...")
        posterior_samples = self.get_chain(discard=self.burn_in, thin=self.thinning, flat=True)
        print(f"Number of effective samples after burn-in and thinning: {len(posterior_samples)}")
        return posterior_samples

    def load_and_plot_observations(self, _dir: str='observations.jpg', title: str = 'Combined Observational Light Curve', x_label: str = 'Phase', y_label: str = 'Magnitude Change', invert_x=False, invert_y=True):
        """
        Loads observational data from specified CSV files, converts flux to magnitude change,
        calculates the orbital phase, and generates a plot of the combined data.

        Returns:
            np.array: A numpy array containing the combined and processed observational
                      data with columns for phase and magnitude change.
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
            sys.exit()

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

                plt.plot(dataset_phases, dataset_mag_changes, 'o', markersize=3, color=color, label=label, alpha=0.7)
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

    @staticmethod
    def log_prior(params):
        """
        Calculates the log-prior probability for the given parameters.
        This is where you define the valid ranges and dynamic dependencies.

        MCMC Parameters (11 total):
        params[0]: exoplanet_orbit_eccentricity
        params[1]: exoplanet_orbit_inclination (deg)
        params[2]: exoplanet_longitude_of_ascending_node (deg)
        params[3]: exoplanet_argument_of_periapsis (deg)
        params[4]: exoplanet_radius (km)
        params[5]: eccentricity (ring)
        params[6]: sma (ring, km) - DEPENDENT
        params[7]: width (ring, km) - DEPENDENT
        params[8]: obliquity (ring, deg)
        params[9]: azimuthal_angle (ring, deg)
        params[10]: argument_of_periapsis (ring, deg)

        Returns:
            float: The log-prior value. Returns -np.inf if parameters are out of bounds.
        """
        global invalid

        exoplanet_orbit_eccentricity, exoplanet_orbit_inclination, \
            exoplanet_longitude_of_ascending_node, exoplanet_argument_of_periapsis, \
            exoplanet_radius, eccentricity, sma, width, obliquity, azimuthal_angle, \
            argument_of_periapsis = params

        density = roche_density(fixed_exoplanet_mass, sma, eccentricity)
        exoplanet_volume = volume(exoplanet_radius)
        exoplanet_density = fixed_exoplanet_mass / exoplanet_volume

        # Check if parameters are within their hard-coded default boundaries
        param_checks = {
            'exoplanet_orbit_eccentricity': exoplanet_orbit_eccentricity,
            'exoplanet_orbit_inclination': exoplanet_orbit_inclination,
            'exoplanet_longitude_of_ascending_node': exoplanet_longitude_of_ascending_node,
            'exoplanet_argument_of_periapsis': exoplanet_argument_of_periapsis,
            'exoplanet_radius': exoplanet_radius,
            'eccentricity': eccentricity,
            'obliquity': obliquity,
            'azimuthal_angle': azimuthal_angle,
            'argument_of_periapsis': argument_of_periapsis
        }

        for name, value in param_checks.items():
            min_bound = specific_parameter_boundaries[name][0]
            max_bound = specific_parameter_boundaries[name][1]
            if not (min_bound <= value <= max_bound):
                # This check is useful for debugging but can be removed for performance
                print(f'Warning: parameter {name} out of boundaries.')
                return -np.inf

        # Dynamic boundaries for dependent parameters
        a_min = exoplanet_radius
        a_max = roche_sma_max(exoplanet_radius, exoplanet_density, eccentricity, density)
        if not (a_min <= sma <= a_max):
            print(f'Warning: parameter semi-major axis out of boundaries.')
            return -np.inf

        w_min = 0
        w_max = a_max - sma
        if not (w_min <= width <= w_max):
            print(f'Warning: parameter width out of boundaries.')
            return -np.inf

        return 0.0

    def log_likelihood(self, params):
        """
        Calculates the log-likelihood of the observed_data given the model parameters.

        Returns:
            float: The log-likelihood value. Returns -np.inf if calculation fails or is invalid.
        """

        exoplanet_orbit_eccentricity, exoplanet_orbit_inclination, \
            exoplanet_longitude_of_ascending_node, exoplanet_argument_of_periapsis, \
            exoplanet_radius, eccentricity, sma, width, obliquity, azimuthal_angle, \
            argument_of_periapsis = params

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
        print(model_output)

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

        if sigma2 == 0: return -np.inf

        log_likelihood_value = -0.5 * np.sum((residuals ** 2) / sigma2 + np.log(2 * np.pi * sigma2))

        if not np.isfinite(log_likelihood_value):
            return -np.inf
        print(log_likelihood_value)
        return log_likelihood_value

    def log_posterior(self, params):
        """
        Calculates the combined log-posterior probability (log-prior + log-likelihood).
        """
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            print('Warning: non-finite log-prior!')
            return -np.inf

        ll = self.log_likelihood(params)
        if not np.isfinite(ll):
            print('Warning: non-finite log-likelihood!')
            return -np.inf

        print(lp + ll)
        return lp + ll

    def trace_plots(self, filename: str = 'trace_plots.png'):
        print("Generating trace plots...")
        fig, axes = plt.subplots(self.ndim, figsize=(12, 2 * self.ndim), sharex=True)
        _samples = self.get_display_samples(raw_chain=True)
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(_samples[:, :, i], "k", alpha=0.3, lw=0.5)
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[0].set_title("MCMC Chains")
        axes[-1].set_xlabel("Iteration")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        print(f"Trace plots saved to {filename}")

    def posterior_diagrams(self, filename: str = 'posterior_histograms.png'):
        print("Generating posterior histograms...")
        fig_hist, axes_hist = plt.subplots(self.ndim, figsize=(10, 2 * self.ndim))
        for i in range(self.ndim):
            sns.histplot(self.get_display_samples()[:, i], kde=True, ax=axes_hist[i], color='skyblue')
            axes_hist[i].set_title(f'Posterior Distribution of {self.labels[i]}')
            axes_hist[i].set_xlabel(self.labels[i])
            axes_hist[i].set_ylabel('Density')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig_hist)
        print(f"Posterior histograms saved to {filename}")

    def summary(self, q1=16, q2=50, q3=84):
        posterior_samples = self.get_display_samples()
        ans = ''
        for i in range(self.ndim):
            mean_val = np.mean(posterior_samples[:, i])
            std_val = np.std(posterior_samples[:, i])
            percentiles = np.percentile(posterior_samples[:, i], [q1, q2, q3])
            ans += f"{self.labels[i]:<28}: {percentiles[1]:.3f} +{percentiles[2] - percentiles[1]:.3f} / -{percentiles[1] - percentiles[0]:.3f}\n"
        return ans

    def get_log_prob(self, **kwargs):
        return super().get_log_prob(discard=self.burn_in, thin=self.thinning, flat=True, **kwargs)

    def best_fit(self):
        log_prob_values = self.get_log_prob()
        best_params_idx = np.argmax(log_prob_values)
        return self.get_effective_samples()[best_params_idx]

    def best_fit_vs_observations(self, filename: str = 'best_fit_model_vs_observed_data.png', title: str = 'Best-Fit Model vs. Observed Data', x_label: str = 'Phase', y_label: str = 'Magnitude Change', invert_x=False, invert_y=True):
        print("\nGenerating best-fit model vs. observed data plot...")
        best_fit_params = self.best_fit()

        best_fit_model_output, transit_duration, _ = calculate_data(fixed_exoplanet_sma,
                                                     best_fit_params[0],
                                                     best_fit_params[1],
                                                     best_fit_params[2],
                                                     best_fit_params[3],
                                                     best_fit_params[4],
                                                     fixed_exoplanet_mass,
                                                     roche_density(fixed_exoplanet_mass, best_fit_params[6], best_fit_params[5]),
                                                     best_fit_params[5],
                                                     best_fit_params[6],
                                                     best_fit_params[7],
                                                     np.nan,
                                                     best_fit_params[8],
                                                     best_fit_params[9],
                                                     best_fit_params[10],
                                                     fixed_specific_absorption_coefficient,
                                                     fixed_star_object,
                                                     fixed_pixel_size,
                                                     custom_units=False
        )

        best_fit_model_lightcurve = np.array(best_fit_model_output)

        observed_phases = (self.observations[:, 0] - 0.5) * 0.5 * fixed_observations_duration / transit_duration + 0.5
        observed_magnitudes = self.observations[:, 1]
        min_phase = np.maximum(np.min(observed_phases), np.min(best_fit_model_lightcurve[:, 0]))
        max_phase = np.minimum(np.max(observed_phases), np.max(best_fit_model_lightcurve[:, 0]))
        intersection = (observed_phases >= min_phase) & (observed_phases <= max_phase)
        obs_phase_intersect = observed_phases[intersection]
        obs_mag_intersect = observed_magnitudes[intersection]

        plt.figure(figsize=(12, 7))
        plt.plot(obs_phase_intersect,obs_mag_intersect, 'o', color='gray', markersize=3,
                 label='Observed Data', alpha=0.5)
        plt.plot(best_fit_model_lightcurve[:, 0], best_fit_model_lightcurve[:, 1], '-', color='red', linewidth=2,
                 label='Best-Fit Model')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        if invert_x:
            plt.gca().invert_xaxis()
        if invert_y:
            plt.gca().invert_yaxis()
        plt.savefig(filename)
        plt.close()
        print(f"Best-fit model vs. observed data plot saved to {filename}")

    def corner_plot(self, filename: str = 'corner_plot.png'):
        fig_corner = corner.corner(
            self.get_display_samples(),
            labels=self.labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 10},
            fill_contours=True,
            smooth=True,
        )
        plt.savefig(filename)
        plt.close(fig_corner)
        print(f"Corner plot saved to {filename}")

class SingleParameterMCMC(MCMC):
    def __init__(self, vary_index: int, nwalkers: int, iterations: int, burn_in: int, thinning: int, files: dict, labels: list[str], initial_guesses: Union[list, np.ndarray]):
        self.vary_index = vary_index
        self.fixed_values = initial_guesses

        super().__init__(nwalkers, 1, iterations, burn_in, thinning, files, labels, initial_guesses[vary_index])

    def log_posterior(self, params):
        val = self.fixed_values
        val[self.vary_index] = params[0]
        return super().log_posterior(val)

fixed_pixel_size = defaults['pixel_size']

# Star parameters
fixed_star_radius = 1.28 * sun_radius # Grouffal et al., 2022
fixed_star_log_g = 4.3 # from TICv8
fixed_star_coefficients = [0.0678, 0.188] # Grant & Wakeford, 2024

fixed_observations_duration = 80 * hrs

# Create the fixed Star object once
fixed_star_object = CustomStarModel(quadratic_star_model, fixed_star_radius, fixed_star_log_g, fixed_star_coefficients, fixed_pixel_size)

# Other fixed parameters (not part of the MCMC parameters)
fixed_exoplanet_mass = 12 * earth_mass  # Santerne et al., 2019
fixed_exoplanet_period = 542.08 * days  # Santerne et al., 2019
fixed_exoplanet_sma = 1.377 * au

fixed_specific_absorption_coefficient = defaults['specific_absorption_coefficient']

guessed_parameter_values = np.array([
    0.002,  # exoplanet orbit eccentricity from Santerne et al., 2019
    89.929 * deg,  # exoplanet orbit inclination from Santerne et al., 2019
    90. * deg,  # exoplanet longitude of ascending node, Grouffal et al., 2025
    0.081 * deg,  # exoplanet argument of periapsis, no available data,
    45000 * km,  # exoplanet radius less than in Santerne et al., 2019
    0.358,  # ring eccentricity, no available data
    90000 * km,  # ring semi-major axis, no available data
    10000 * km,  # ring width, no available data
    87.012 * deg,  # ring obliquity, maximum optical depth
    0.082 * deg,  # ring azimuthal angle, maximum optical depth
    0.088 * deg  # ring argument of periapsis, no available data
])

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    mcmc1 = SingleParameterMCMC(vary_index=5, nwalkers=10, iterations=5000, burn_in=500, thinning=1, # select element index to vary
                   files={
                            "C18 short cadence": ("observations/C18_short_cadence.csv", "red"),
                            "C18 long cadence": ("observations/C18_long_cadence.csv", "green"),
                            "C5": ("observations/C5.csv", "blue"),
                        },
                   labels=[
                         'Orbit Eccentricity', 'Orbit Inclination, °',
                         'Ex. Long. Asc. Node, °',
                         'Ex. Arg. of Periapsis, °', 'Exoplanet Radius, km', 'Ring Ecccentricity',
                         'Ring Semi-major Axis, km', 'Ring Width, km', 'Ring Obliquity, °',
                         'Ring Azimuthal Angle, °', 'Ring Arg. of Periapsis, °'
                        ],
                   initial_guesses=guessed_parameter_values)

    mcmc = MCMC(ndim=11, nwalkers=50, iterations=10000, burn_in=500, thinning=10,
                   files={
                            "C18 short cadence": ("observations/C18_short_cadence.csv", "red"),
                            "C18 long cadence": ("observations/C18_long_cadence.csv", "green"),
                            "C5": ("observations/C5.csv", "blue"),
                        },
                   labels=[
                         'Orbit Eccentricity', 'Orbit Inclination, °',
                         'Ex. Long. Asc. Node, °',
                         'Ex. Arg. of Periapsis, °', 'Exoplanet Radius, km', 'Ring Ecccentricity',
                         'Ring Semi-major Axis, km', 'Ring Width, km', 'Ring Obliquity, °',
                         'Ring Azimuthal Angle, °', 'Ring Arg. of Periapsis, °'
                        ],
                   initial_guesses=guessed_parameter_values
                )

    mcmc.run()
    mcmc.save_chain()
    mcmc.trace_plots()  # plotting the chains for diagnostics
    mcmc.posterior_diagrams()  # plotting histograms of the posterior distributions
    statistics = mcmc.summary()  # obtaining parameter values
    print(f'Parameter Values:\n{statistics}')
    with open('summary.txt', 'w', encoding='utf-8') as file:
        file.write(statistics)
    mcmc.best_fit_vs_observations()
    mcmc.corner_plot()  # generate corner plot
