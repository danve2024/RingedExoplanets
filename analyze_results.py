import numpy as np
import matplotlib.pyplot as plt
import corner
import os
import sys
import json
import pandas as pd
import seaborn as sns
from dynesty.utils import resample_equal
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec

# Set plot style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_context('talk')
sns.set_palette('viridis')

# Default parameter labels (update these based on your model)
DEFAULT_LABELS = [
    'Orbit Eccentricity',
    'Orbit Inclination (deg)',
    'Long. Asc. Node (deg)',
    'Arg. of Periapsis (deg)',
    'Planet Radius (km)',
    'Ring Eccentricity',
    'Ring Semi-major Axis (km)',
    'Ring Width (km)',
    'Ring Obliquity (deg)',
    'Ring Azimuthal Angle (deg)',
    'Ring Arg. of Periapsis (deg)'
]

def load_results(filename):
    """Load nested sampling results from NPZ file."""
    try:
        data = np.load(filename, allow_pickle=True)
        
        # Handle both .npz and .npy files
        if filename.endswith('.npz'):
            results = {
                'samples': data['samples'],
                'logwt': data['logwt'],
                'logz': data['logz'],
                'logl': data['logl']
            }
            if 'labels' in data:
                results['labels'] = data['labels']
            else:
                n_params = data['samples'].shape[1]
                results['labels'] = DEFAULT_LABELS[:n_params]
        else:
            # Handle raw .npy file
            samples = data
            results = {
                'samples': samples,
                'logwt': np.ones(len(samples)),
                'logz': np.array([0.0]),
                'logl': np.zeros(len(samples)),
                'labels': DEFAULT_LABELS[:samples.shape[1]]
            }
        
        return results
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None


def get_posterior_samples(results, nsamples=1000):
    """Get equally weighted posterior samples."""
    # Get weights (normalized to sum to 1)
    weights = np.exp(results['logwt'] - results['logz'][-1])
    weights /= np.sum(weights)  # Ensure weights sum to 1
    
    # Resample to get equally weighted samples
    samples = resample_equal(
        samples=results['samples'],
        weights=weights
    )
    
    # Return the requested number of samples
    return samples[:nsamples] if nsamples else samples


def create_corner_plot(samples, labels, filename='corner_plot.png'):
    """Create a corner plot of the posterior distributions."""
    if samples is None or len(samples) == 0:
        print("No samples available for corner plot.")
        return
    
    # Use a smaller number of points for better visualization
    n_samples = min(5000, len(samples))
    plot_samples = samples[np.random.choice(len(samples), n_samples, replace=False)]
    
    # Create figure with custom settings
    fig = corner.corner(
        plot_samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 12},
        plot_datapoints=True,
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        levels=1.0 - np.exp(-0.5 * np.array([1.0, 2.0])**2),
        color='#0072B2',
        alpha=0.5,
        bins=30,
        smooth=1.0,
        smooth1d=1.0,
        hist_kwargs={'density': True, 'color': '#4D8DC4', 'histtype': 'stepfilled'}
    )
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
    
    # Add title
    fig.suptitle('Parameter Posterior Distributions', y=1.02, fontsize=16)
    
    # Save figure
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Corner plot saved to {filename}")


def plot_correlation_heatmap(samples, labels, filename='correlation_heatmap.png'):
    """Create a correlation heatmap of the parameters."""
    print(f"Creating correlation heatmap: {filename}")
    
    # Calculate correlation matrix
    corr = np.corrcoef(samples.T)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        xticklabels=labels,
        yticklabels=labels
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Correlation heatmap saved to {filename}")


def print_summary(samples, labels):
    """Print summary statistics for the parameters."""
    if samples is None or len(samples) == 0:
        print("No samples available for summary statistics.")
        return
    
    print("\nParameter estimates (50th percentile) with 1-sigma uncertainties:")
    print("-" * 70)
    
    for i, label in enumerate(labels):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{label:<30}: {q50:.3f} +{q84 - q50:.3f}/-{q50 - q16:.3f}")


def plot_marginal_distributions(samples, labels, filename='marginal_distributions.png'):
    """Create marginal distributions for each parameter."""
    print(f"Creating marginal distributions plot: {filename}")
    
    n_params = samples.shape[1]
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.ravel()
    
    for i in range(n_params):
        ax = axes[i]
        param = samples[:, i]
        
        # Kernel density estimation
        kde = gaussian_kde(param)
        x = np.linspace(np.min(param), np.max(param), 1000)
        
        # Plot histogram and KDE
        sns.histplot(param, ax=ax, stat='density', alpha=0.3, color='#4D8DC4')
        ax.plot(x, kde(x), color='#0072B2', lw=2)
        
        # Add vertical lines for percentiles
        q16, q50, q84 = np.percentile(param, [16, 50, 84])
        ax.axvline(q50, color='#D55E00', linestyle='-', lw=1.5, label='Median')
        ax.axvline(q16, color='#D55E00', linestyle='--', lw=1, label='1σ')
        ax.axvline(q84, color='#D55E00', linestyle='--', lw=1)
        
        # Add text with parameter values
        text = f"{q50:.3f} +{q84-q50:.3f}\n    -{q50-q16:.3f}"
        ax.text(0.05, 0.9, text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(labels[i])
        ax.set_xlabel('Parameter Value')
        ax.legend()
    
    # Remove empty subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Marginal distributions plot saved to {filename}")


def plot_trace_plots(samples, labels, filename='trace_plots.png'):
    """Create trace plots to assess convergence."""
    print(f"Creating trace plots: {filename}")
    
    n_params = samples.shape[1]
    n_cols = min(2, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows))
    axes = axes.ravel()
    
    for i in range(n_params):
        ax = axes[i]
        ax.plot(samples[:, i], alpha=0.7, color='#4D8DC4')
        ax.set_title(labels[i])
        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Parameter Value')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Trace plots saved to {filename}")


def save_parameter_summary(samples, labels, filename='parameter_summary.txt'):
    """Save detailed parameter statistics to a file."""
    with open(filename, 'w') as f:
        # Header
        f.write("Parameter Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic statistics
        f.write("Parameter Estimates (50th percentile) with 1-sigma uncertainties:\n")
        f.write("-" * 80 + "\n")
        
        for i, label in enumerate(labels):
            q05, q16, q50, q84, q95 = np.percentile(samples[:, i], [5, 16, 50, 84, 95])
            f.write(f"{label}:\n")
            f.write(f"  Median: {q50:.6f}\n")
            f.write(f"  68% CI: [{q16:.6f}, {q84:.6f}]\n")
            f.write(f"  90% CI: [{q05:.6f}, {q95:.6f}]\n")
            f.write(f"  Range:  [{np.min(samples[:, i]):.6f}, {np.max(samples[:, i]):.6f}]\n\n")
        
        # Correlation matrix
        f.write("\nCorrelation Matrix:\n")
        f.write("-" * 80 + "\n")
        
        corr = np.corrcoef(samples.T)
        n_params = len(labels)
        
        # Write header
        f.write(" " * 15)
        for i in range(n_params):
            f.write(f"{i+1:>8d}")
        f.write("\n" + "-" * (15 + 8 * n_params) + "\n")
        
        # Write rows
        for i in range(n_params):
            f.write(f"{labels[i][:12]:<15}")
            for j in range(n_params):
                if j < i:
                    f.write(" " * 8)
                else:
                    f.write(f"{corr[i, j]:8.3f}")
            f.write("\n")
    
    print(f"Parameter summary saved to {filename}")


def save_best_fit_parameters(samples, labels, filename='best_fit_params.json'):
    """Save best-fit parameters to a JSON file."""
    best_fit = {}
    
    for i, label in enumerate(labels):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        best_fit[label] = {
            'median': float(q50),
            'lower_1sigma': float(q16),
            'upper_1sigma': float(q84),
            'uncertainty': float((q84 - q16) / 2)
        }
    
    with open(filename, 'w') as f:
        json.dump(best_fit, f, indent=4)
    
    print(f"Best-fit parameters saved to {filename}")
    return best_fit


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze nested sampling results')
    parser.add_argument('input', nargs='?', default='nested_sampling_results.npz',
                      help='Input file with sampling results (default: nested_sampling_results.npz)')
    parser.add_argument('--output', '-o', default='analysis_results',
                      help='Output directory for results (default: analysis_results)')
    parser.add_argument('--nsamples', '-n', type=int, default=10000,
                      help='Number of posterior samples to use (default: 10000)')
    parser.add_argument('--thin', type=int, default=1,
                      help='Thinning factor for trace plots (default: 1)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    
    if results is None:
        print("Failed to load results. Please check the input file.")
        return
    
    # Get parameter labels
    if 'labels' in results:
        labels = results['labels']
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
    else:
        n_params = results['samples'].shape[1]
        labels = [f'Parameter {i+1}' for i in range(n_params)]
    
    # Get posterior samples
    print("\nGenerating posterior samples...")
    samples = get_posterior_samples(results, args.nsamples)
    
    if samples is None or len(samples) == 0:
        print("Error: No samples generated. Check your input file.")
        return
    
    print(f"\nAnalyzing {len(samples)} samples with {samples.shape[1]} parameters...")
    
    # Create plots and save results
    print("\nCreating visualizations...")
    
    # 1. Save parameter summary
    summary_file = os.path.join(args.output, 'parameter_summary.txt')
    save_parameter_summary(samples, labels, filename=summary_file)
    
    # 2. Save best-fit parameters
    best_fit_file = os.path.join(args.output, 'best_fit_parameters.json')
    best_fit = save_best_fit_parameters(samples, labels, filename=best_fit_file)
    
    # 3. Create corner plot
    corner_plot = os.path.join(args.output, 'corner_plot.png')
    plot_corner(samples, labels, filename=corner_plot)
    
    # 4. Create correlation heatmap
    heatmap = os.path.join(args.output, 'correlation_heatmap.png')
    plot_correlation_heatmap(samples, labels, filename=heatmap)
    
    # 5. Create marginal distributions
    marginals = os.path.join(args.output, 'marginal_distributions.png')
    plot_marginal_distributions(samples, labels, filename=marginals)
    
    # 6. Create trace plots (thinned for better visualization)
    if args.thin > 1:
        trace_samples = samples[::args.thin]
    else:
        trace_samples = samples
    trace_plot = os.path.join(args.output, 'trace_plots.png')
    plot_trace_plots(trace_samples, labels, filename=trace_plot)
    
    # Save the processed samples
    samples_file = os.path.join(args.output, 'posterior_samples.npy')
    np.save(samples_file, samples)
    
    print("\n" + "="*80)
    print("Analysis complete! Results saved to:")
    print(f"  - Parameter summary: {summary_file}")
    print(f"  - Best-fit parameters: {best_fit_file}")
    print(f"  - Corner plot: {corner_plot}")
    print(f"  - Correlation heatmap: {heatmap}")
    print(f"  - Marginal distributions: {marginals}")
    print(f"  - Trace plots: {trace_plot}")
    print(f"  - Posterior samples: {samples_file}")
    print("\nBest-fit parameters (median ± 1σ):")
    for param, values in best_fit.items():
        print(f"  {param}: {values['median']:.4f} ± {values['uncertainty']:.4f}")
    print("="*80)

    # Load the results
    try:
        print("Loading results...")
        results = load_results('nested_sampling_results.npy')
        print("Results loaded successfully.")

        # Get posterior samples
        print("Processing samples...")
        samples = get_posterior_samples(results)

        if samples is not None:
            print(f"Successfully processed {len(samples)} samples.")
            
            # Create corner plot
            create_corner_plot(samples, labels)
            
            # Print summary statistics
            print_summary(samples, labels)
            
            # Save samples to file
            np.save('posterior_samples.npy', samples)
            print("\nPosterior samples saved to 'posterior_samples.npy'")
            
        else:
            print("Could not extract samples from results. Available keys/attributes:")
            if hasattr(results, '__dict__'):
                print(results.__dict__.keys())
            elif isinstance(results, dict):
                print(results.keys())
            else:
                print("No dictionary or object attributes found.")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Analysis failed. Please check the error message above.")


if __name__ == "__main__":
    main()