from models import disk, elliptical_ring, _array_to_qimage
import numpy as np
from main import calculate_data, defaults
from visualization import FramesWindow
from space import Star, Void, limb_darkening
from PyQt6.QtGui import QImage
from io import BytesIO
import matplotlib.pyplot as plt
from formulas import roche_density, to_pixels
from units import *
from measure import Measure


def draw_ring(size=401, R=20, a=100, e=0.2, w=4, theta=30, phi=40, psi=0, tau=1.4, infinity=10, constant_fill=False, **kwargs):
    return _array_to_qimage(np.maximum(elliptical_ring(size, a, e, w, theta, phi, psi, tau, constant_fill), disk(R, size, infinity)))

def draw_orbit(size=401, R_S=20, A=100, e_p=0.2, i=30, Omega=40, omega=0, infinity=10, **kwargs):
    return draw_ring(size, R_S, A, e_p, 2, i, Omega, omega, 1, 1, True)

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

def draw_animation(f, gif=False, name='animation', infinity=10, abbr=None, unit='', unit_coefficient=1, suffix=None,  **kwargs):
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

    frames = []
    values = {}
    for _i in space:
        values[param] = _i
        if isinstance(_i, float) or isinstance(_i, int) or isinstance(_i, Measure.Unit) or isinstance(_i, np.floating):
            frames.append(f(**values, **fixed, infinity=infinity, label=f'{abbr}={round(_i, 2) * unit_coefficient}{unit}{suffix}'))
        else:
            frames.append(f(**values, **fixed, infinity=infinity, label=f'{abbr}: {_i}{suffix}'))

    anim = FramesWindow(frames)
    if gif:
        anim.save_gif(name + '.gif')
    else:
        anim.save_frames(name)

def draw_ring_animation(gif=False, name='animation', infinity=10, **kwargs):
    draw_animation(draw_ring, gif, name, infinity, **kwargs)

def draw_orbit_animation(gif=False, name='animation', **kwargs):
    draw_animation(draw_orbit, gif, name, **kwargs)

def draw_lightcurve_animation(gif=False, name='animation', **kwargs):
    draw_animation(draw_lightcurve, gif, name, **kwargs)

draw_ring_animation(gif=True, name='ring_result', infinity=10, R=[to_pixels(58677.6 * km)], fixed_e=0.365, fixed_a=to_pixels(114804 * km), fixed_w=to_pixels(25512 * km), fixed_theta=0.077, fixed_phi=90.099, fixed_psi=0.081)
