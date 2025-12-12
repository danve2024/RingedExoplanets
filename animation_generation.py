from models import disk, elliptical_ring, _array_to_qimage, planet
import numpy as np
from main import calculate_data, defaults
from nested_sampling import fixed_exoplanet_sma, fixed_exoplanet_mass
from visualization import FramesWindow
from space import Star, Orbit, Exoplanet
from PyQt6.QtGui import QImage
from io import BytesIO
import matplotlib.pyplot as plt
from formulas import roche_density, to_pixels
from alternative_models import oblate_planet, spotted_star_transit
from units import *
from measure import Measure


def draw_ring(size=401, R=20, a=100, e=0.2, w=4, theta=30, phi=40, psi=0, tau=1.4, infinity=10, constant_fill=False, **kwargs):
    return _array_to_qimage(np.maximum(elliptical_ring(size, a, e, w, theta, phi, psi, tau, constant_fill), disk(R, size, infinity)))

def draw_orbit(size=401, R_S=20, A=100, e_p=0.2, i=30, Omega=40, omega=0, infinity=10, **kwargs):
    return draw_ring(size, R_S, A, e_p, 2, i, Omega, omega, 1, 1, True)

def draw_planet(size=401, R=100, f=0, rho=0, infinity=1, **kwargs):
    return _array_to_qimage(planet(R, size, infinity, f, rho))

def draw_lightcurve(A=defaults['exoplanet_sma'].min,
                    e_p=defaults['exoplanet_orbit_eccentricity'].mean,
                    i=2,
                    Omega=30,
                    omega=0,
                    R=defaults['exoplanet_radius'].mean,
                    M=1.9e27,
                    rho=None,
                    e=defaults['eccentricity'].mean,
                    a=None,
                    w=None,
                    theta=30,
                    phi=20,
                    psi=0,
                    kappa=defaults['specific_absorption_coefficient'],
                    R_S=defaults['star_radius'],
                    T=defaults['star_temperature'],
                    log_g=defaults['star_log(g)'],
                    wavelength=defaults['wavelength'],
                    band='u',
                    px=defaults['pixel_size'],
                    limb_darkening_model=defaults['limb_darkening'],
                    label='',
                    print_values=True,
                    **kwargs):

    star = Star(R_S, T, log_g, wavelength, band, px, limb_darkening_model)

    if a is None: a = defaults['exoplanet_radius'].mean * 5
    if w is None: w = defaults['exoplanet_radius'].mean / 2
    if rho is None: rho = roche_density(M, a, e)

    data = calculate_data(A, e_p, i, Omega, omega, R, M, rho, e, a, w, np.nan, theta, phi, psi, kappa, star, px, False)[0]
    if print_values:
        print(data)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot all points in blue
    ax.plot([data[i][0] for i in range(len(data))], [data[i][1] for i in range(len(data))], label='model')

    ax.invert_yaxis()
    ax.set_title(f'Transit light curve ({label})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Magnitude Change')
    ax.legend()

    # Convert the Matplotlib figure to a QImage
    # Use a BytesIO object to capture the figure as a PNG
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Close the figure to free up memory
    buf.seek(0)

    # Create a QImage from the PNG data
    image = QImage.fromData(buf.getvalue())
    return image

def draw_oblate_planet_light_curve(A=defaults['exoplanet_sma'].min,
                    e_p=defaults['exoplanet_orbit_eccentricity'].mean,
                    i=2,
                    Omega=30,
                    omega=0,
                    R=defaults['exoplanet_radius'].mean,
                    M=1.9e27,
                    f=0.,
                    r=0,
                    R_S=defaults['star_radius'],
                    T=defaults['star_temperature'],
                    log_g=defaults['star_log(g)'],
                    wavelength=defaults['wavelength'],
                    band='u',
                    px=defaults['pixel_size'],
                    limb_darkening_model=defaults['limb_darkening'],
                    label='',
                    print_values=True,
                    **kwargs):
    star = Star(R_S, T, log_g, wavelength, band, px, limb_darkening_model)

    data = oblate_planet(A, e_p, i, Omega, omega, R, M, f, r, star, px, False)[0]
    if print_values:
        print(data)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot all points in blue
    ax.plot([data[i][0] for i in range(len(data))], [data[i][1] for i in range(len(data))], label='model')

    ax.invert_yaxis()
    ax.set_title(f'Transit light curve ({label})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Magnitude Change')
    ax.legend()

    # Convert the Matplotlib figure to a QImage
    # Use a BytesIO object to capture the figure as a PNG
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Close the figure to free up memory
    buf.seek(0)

    # Create a QImage from the PNG data
    image = QImage.fromData(buf.getvalue())
    return image


def draw_spotted_star_light_curve(
        A=defaults['exoplanet_sma'].min,
        e_p=defaults['exoplanet_orbit_eccentricity'].mean,
        i=2,
        Omega=30,
        omega=0,
        R=defaults['exoplanet_radius'].mean,
        M=1.9e27,
        R_S=defaults['star_radius'],
        T=defaults['star_temperature'],
        log_g=defaults['star_log(g)'],
        wavelength=defaults['wavelength'],
        band='u',
        px=defaults['pixel_size'],
        limb_darkening_model=defaults['limb_darkening'],
        angular_velocity=1,
        spot_longitude=90,
        spot_radius=0.1,
        spot_brightness=0.7,
        label='',
        print_values=True,
        **kwargs
):
    """
    Generate a light curve of a planet transiting a spotted star.

    Args match the standard light curve functions but add spot parameters.
    """
    # Create star with spots
    star = Star(R_S, T, log_g, wavelength, band, px, limb_darkening_model)
    
    # Calculate data with spots
    data = spotted_star_transit(A, e_p, i, Omega, omega, R, M, R_S, T, log_g, wavelength, px, angular_velocity, spot_longitude, spot_radius, spot_brightness)[0]

    if print_values:
        print(data)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot all points in blue, matching draw_lightcurve style
    ax.plot([data[i][0] for i in range(len(data))], 
            [data[i][1] for i in range(len(data))], 
            label='model')

    ax.invert_yaxis()
    ax.set_title(f'Transit light curve ({label})')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Magnitude Change')
    ax.legend()

    # Convert to QImage
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return QImage.fromData(buf.getvalue())


def draw_animation(fn, gif=False, name='animation', infinity=10, abbr=None, unit='', unit_coefficient=1, suffix=None,  **kwargs):
    if suffix is None:
        suffix = ''
    else:
        suffix = ', ' + suffix
    try:
        param = list(kwargs.keys())[0]
    except IndexError:
        raise ValueError('Nothing to do!')
    if abbr is None:
        abbr = param
    fixed = {}
    for i in kwargs:
        if i.startswith('fixed_'):
            fixed[i.replace('fixed_', '')] = kwargs[i]
    rng = kwargs[param]
    if isinstance(rng, tuple):
        start = rng[0]
        end = rng[1]
        num = rng[2]
        space = np.linspace(start, end, num)
    elif isinstance(rng, list):
        space = rng
    else:
        space = []

    # Create frames with their corresponding values for sorting
    frames_with_values = []
    values = {}
    for _i in space:
        values[param] = _i
        if isinstance(_i, float) or isinstance(_i, int) or isinstance(_i, Measure.Unit) or isinstance(_i, np.floating):
            frame = fn(**values, **fixed, infinity=infinity, label=f'{abbr}={round(_i, 2) * unit_coefficient}{unit}{suffix}')
            frames_with_values.append((_i, frame))
        else:
            frame = fn(**values, **fixed, infinity=infinity, label=f'{abbr}: {_i}{suffix}')
            frames_with_values.append((_i, frame))
    
    # Sort frames by their value in ascending order
    frames_with_values.sort(key=lambda x: x[0] if not isinstance(x[0], str) else float('inf'))
    
    # Extract just the frames in sorted order
    frames = [frame for _, frame in frames_with_values]

    anim = FramesWindow(frames)
    if gif:
        anim.save_gif(f'./{name}.gif')
    else:
        anim.save_frames(name)

def draw_ring_animation(gif=False, name='animation', infinity=10, **kwargs):
    draw_animation(draw_ring, gif, name, infinity, **kwargs)

def draw_orbit_animation(gif=False, name='animation', **kwargs):
    draw_animation(draw_orbit, gif, name, **kwargs)

def draw_light_curve_animation(gif=False, name='animation', **kwargs):
    draw_animation(draw_lightcurve, gif, name, **kwargs)

def draw_oblate_planet_light_curve_animation(gif=False, name='animation', **kwargs):
    draw_animation(draw_oblate_planet_light_curve, gif, name, **kwargs)

def draw_planet_animation(gif=False, name='animation', **kwargs):
    draw_animation(draw_planet, gif, name, **kwargs)

def draw_spotted_star_light_curve_animation(gif=False, name='animation', **kwargs):
    draw_animation(draw_spotted_star_light_curve, gif, name, **kwargs)
