# Ringed Exoplanets Transit Modeling

This project provides a comprehensive framework for modeling and analyzing transits of ringed exoplanets, with a focus on HIP 41378f. The codebase includes physical models, parameter estimation methods (MCMC and nested sampling), visualization tools, and analysis utilities.

## Main File

**`nested_sampling.py`** - This is the main file for performing nested sampling analysis on HIP 41378f. It implements dynamic nested sampling using the `dynesty` library to estimate posterior distributions of exoplanet and ring system parameters. The file includes:

- `NestedSampler` class for running nested sampling analysis
- Fixed parameters specific to HIP 41378f (star properties, exoplanet mass, orbital period, etc.)
- Parameter boundaries optimized for HIP 41378f analysis
- Methods for saving results and generating analysis plots

### Main Ringed Planet Model (`nested_sampling.py`)

**Fixed Parameters:**
- Star radius: 1.28 × solar radius (Grouffal et al., 2022)
- Star log(g): 4.3 (from TICv8)
- Star limb darkening coefficients: [0.0678, 0.188] (Grant & Wakeford, 2024)
- Exoplanet mass: 12 × Earth mass (Santerne et al., 2019)
- Exoplanet orbital period: 542.08 days (Santerne et al., 2019)
- Exoplanet semi-major axis: 1.377 AU
- Observation duration: 80 hours
- Pixel size: From defaults
- Specific absorption coefficient: From defaults

**Parameter Boundaries (11 dimensions):**
- Exoplanet orbit eccentricity: (0, 0.4)
- Exoplanet orbit inclination: (0°, 90°)
- Exoplanet longitude of ascending node: (90° - 5e-8, 90° + 5e-8)
- Exoplanet argument of periapsis: (0°, 180°)
- Exoplanet radius: (2,000 km, 58,880 km) [9.2 × Earth radius]
- Ring eccentricity: (0, 0.9)
- Ring semi-major axis: (dynamic, based on Roche limit)
- Ring width: (0, dynamic maximum based on Roche limit)
- Ring obliquity: (0°, 90°)
- Ring azimuthal angle: (0°, 180°)
- Ring argument of periapsis: (0°, 180°)

### Noise Model (`noise_nested_sampling.py`)

**Parameter Boundaries (7 dimensions):**
- Exoplanet orbit eccentricity: (0, 0.4)
- Exoplanet orbit inclination: (0°, 90°)
- Exoplanet longitude of ascending node: (90° - 5e-8, 90° + 5e-8)
- Exoplanet argument of periapsis: (0°, 180°)
- Exoplanet radius: (2,000 km, 128,000 km)
- Noise scale: (0, auto-calculated from data)
- Noise magnitude: (0, auto-calculated from data)

### Oblate Planet Model (`oblate_planet_nested_sampling.py`)

**Parameter Boundaries (7 dimensions):**
- Exoplanet orbit eccentricity: (0, 0.4)
- Exoplanet orbit inclination: (0°, 90°)
- Exoplanet longitude of ascending node: (90° - 5e-8, 90° + 5e-8)
- Exoplanet argument of periapsis: (0°, 180°)
- Exoplanet radius: (2,000 km, 128,000 km)
- Planet oblateness: (0, 1)
- Projection rotation: (0°, 180°)

### Ringless Model (`ringless_nested_sampling.py`)

**Parameter Boundaries (5 dimensions):**
- Exoplanet orbit eccentricity: (0, 0.4)
- Exoplanet orbit inclination: (0°, 90°)
- Exoplanet longitude of ascending node: (90° - 5e-8, 90° + 5e-8)
- Exoplanet argument of periapsis: (0°, 180°)
- Exoplanet radius: (2,000 km, 128,000 km)

### Star Spots Model (`star_spots_nested_sampling.py`)

**Parameter Boundaries (9 dimensions):**
- Exoplanet orbit eccentricity: (0, 0.4)
- Exoplanet orbit inclination: (0°, 90°)
- Exoplanet longitude of ascending node: (90° - 5e-8, 90° + 5e-8)
- Exoplanet argument of periapsis: (0°, 180°)
- Exoplanet radius: (2,000 km, 128,880 km)
- Star angular velocity: (0, 1800 °/day)
- Spot longitude: (0°, 360°)
- Spot radius: (0, 1) [in star radii]
- Spot brightness: (0, 2) [relative to star]

**Common Observation Files (for all models):**
- C18 short cadence: `observations/C18_short_cadence.csv`
- C18 long cadence: `observations/C18_long_cadence.csv`
- C5: `observations/C5.csv`

## File Descriptions

### Core Physics and Models

- **`space.py`** - Defines the main physical objects: `Star`, `Exoplanet`, `Orbit`, `Rings`, `StarSpot`, and `CustomStarModel`. Handles star limb darkening models, exoplanet ring systems, and orbital mechanics. Includes a demonstration script for star spots animation.

- **`models.py`** - Contains functions for generating physical models:
  - `planet()` - Creates 2D projections of planets with oblateness and rotation
  - `disk()` - Creates circular disk models
  - `transit()` - Calculates transit light curves based on orbital mechanics
  - `transit_animation()` - Generates animation frames for transit visualization
  - `star_spot()` - Models stellar spots with rotation
  - `quadratic_star_model()` and `square_root_star_model()` - Limb darkening models

- **`formulas.py`** - Mathematical formulas and physical calculations:
  - Roche limit calculations (minimum/maximum ring semi-major axis)
  - Hill sphere calculations
  - Orbital mechanics (mean anomaly, eccentric anomaly, true anomaly, radius vector)
  - Star mass calculations
  - Unit conversions (to_pixels)
  - Light curve formatting

- **`units.py`** - Physical unit definitions and conversions using the `Measure` class. Defines astronomical units (AU, km, kg, etc.), time units, and physical constants.

- **`measure.py`** - Implements the `Measure` class for handling physical quantities with units, including unit conversions and arithmetic operations.

### Main Application

- **`main.py`** - Main application entry point. Defines default parameter ranges and the `calculate_data()` function that orchestrates the simulation by creating rings, orbits, exoplanets, and computing transit light curves. Can be run as a GUI application for interactive parameter exploration.

### Parameter Estimation

- **`data_fitting.py`** - Implements MCMC (Markov Chain Monte Carlo) parameter estimation using `emcee`:
  - `MCMC` class extending `emcee.EnsembleSampler`
  - Log-likelihood and log-prior calculations
  - Chain analysis and visualization (corner plots, trace plots, posterior histograms)
  - Correlation heatmaps

- **`nested_sampling.py`** - **Main file** for nested sampling analysis (see above for details)

- **`complete_nested_sampling_analysis.py`** - Extended analysis tools for nested sampling results, including additional visualization and statistical analysis methods.

### Data Handling

- **`observations.py`** - `Observations` class for loading and processing observation data from CSV files. Handles time and magnitude shifts, normalization, and data manipulation for comparison with models.


### Visualization

- **`visualization.py`** - PyQt6-based GUI components:
  - `Selection` - Parameter selection interface
  - `Model` - Main visualization window for transit models
  - `FramesWindow` - Animation display window
  - `AnimatedGraph` - Animated graph visualization
  - `AnimationWindow` - General animation window

- **`animation_generation.py`** - Functions for generating animations and visualizations of parameter effects, including light curve animations and ring parameter demonstrations.

- **`plot_parameters.py`** - Utilities for plotting parameter distributions and relationships.

### Analysis and Results

- **`analyze_results.py`** - Analysis tools for examining MCMC and nested sampling results, including statistical summaries and comparisons.

- **`kfold.py`** - K-fold cross-validation implementation for model validation.

- **`alternative_models.py`** - Alternative model implementations for comparison studies.

### Data and Configuration

- **`limb_darkening/`** - Directory containing limb darkening coefficient data:
  - `quadratic.json` - Quadratic limb darkening coefficients
  - `square-root.json` - Square-root limb darkening coefficients

- **`observations/`** - Directory containing observation data files in CSV format

- **`nested_sampling/`** - Directory for nested sampling log-likelihood cache and results

## Usage

### Running Nested Sampling Analysis (HIP 41378f)

```python
python nested_sampling.py
```

This will:
1. Initialize the nested sampler with HIP 41378f parameters
2. Run the nested sampling analysis
3. Save results to `nested_sampling_results.npz`
4. Generate analysis plots and summary statistics

### Running MCMC Analysis

```python
python data_fitting.py
```

### Running Interactive GUI

```python
python main.py
```

## Output Files

- **`nested_sampling_results.npz`** - Nested sampling results (samples, weights, log-likelihoods)
- **`nested_sampling_analysis_summary.txt`** - Text summary of analysis results
- **`corner_plot.png`** - Corner plot showing parameter posterior distributions
- **`best_fit_model_vs_observed_data.png`** - Comparison of best-fit model with observations
- **`parameter_distributions.png`** - Parameter distribution plots
- **`trace_plots.png`** - MCMC chain trace plots (for MCMC runs)

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Note: The project uses `dynesty` for nested sampling, which should be installed separately if not included in requirements.txt.

## References

- Grouffal et al., 2022 - Star radius for HIP 41378f
- Santerne et al., 2019 - Exoplanet mass and orbital period
- Grant & Wakeford, 2024 - Limb darkening coefficients
- TICv8 - Stellar log(g) value

