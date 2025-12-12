import math
import time
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QVBoxLayout, QDialog, QLabel
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

from formulas import light_curve, mean_anomaly, eccentric_anomaly, true_anomaly, radius_vector

"""
Uses numpy arrays to model the event of the transit

0 - minimum matrix pixel brightness
1 - maximum matrix pixel brightness
"""

def planet(radius: float, size: float = None, fill: float = 1.0, oblateness: float = 0.0, rotation_angle: float = 0.0) -> np.ndarray:
    """
    Creates a 2D projection of a planet, which can be spherical or oblate.

    :param radius: The equatorial radius of the planet in pixels (χ_R)
    :param size: The size of the output array (χ_a). If None, calculated automatically.
    :param fill: Value to fill the planet with (e.g., optical depth)
    :param oblateness: Flattening factor (0 = sphere, ~0.3 = highly oblate)
    :param rotation_angle: Rotation angle in degrees (0-360)
    :return: numpy array with the planet model
    """
    if size is None:
        size = 2 * radius + 1
    size = int(round(size))
    if size == 0:
        return np.zeros((1, 1))
    if size % 2 == 0:
        size -= 1

    center = (size - 1) / 2.0
    y, x = np.ogrid[:size, :size]

    # Shift coordinates to be centered
    x_centered = x - center
    y_centered = y - center

    # Apply rotation if needed
    if rotation_angle != 0:
        theta = np.radians(-rotation_angle)
        x_rot = x_centered * np.cos(theta) - y_centered * np.sin(theta)
        y_rot = x_centered * np.sin(theta) + y_centered * np.cos(theta)
        x_centered, y_centered = x_rot, y_rot

    # Create the planet model
    model = np.zeros((size, size), dtype=float)

    # Calculate semi-axes for the ellipse
    a = radius  # Semi-major axis (equatorial radius)
    b = radius * (1 - oblateness)  # Semi-minor axis (polar radius)

    # Ellipse equation: (x/a)² + (y/b)² ≤ 1
    if oblateness != 0:
        mask = (x_centered**2 / a**2) + (y_centered**2 / b**2) <= 1.0
    else:  # For perfect sphere, use circle equation for better performance
        mask = (x_centered**2 + y_centered**2) <= radius**2

    model[mask] = fill
    return model

disk = planet

def elliptical_ring(
        size: int,
        a: float,
        e: float,
        w: float,
        obliquity: float,
        azimuthal_angle: float,
        argument_of_periapsis: float,
        fill: float,
        constant_fill=False) -> np.array:
    """
    See equations 2.2.3 - 2.2.26.
    Draws an elliptical ring using standard orbital parameters for orientation.

    :param size: χ_a
    :param a: α
    :param e: e
    :param w: χ_w
    :param obliquity: θ
    :param azimuthal_angle: φ
    :param argument_of_periapsis: ψ
    :param fill: τ
    :return: numpy array with an elliptical ring of specified parameters
    """

    size = int(round(size))
    if size % 2 == 0:
        size -= 1

    inclination_rad = np.deg2rad(90.0 - obliquity) # υ
    azimuthal_angle_rad = np.deg2rad(azimuthal_angle)
    arg_peri_rad = np.deg2rad(argument_of_periapsis)

    a_outer = a + w / 2.0 # α_outer
    a_inner = a - w / 2.0 # α_inner

    # Rotational matrices
    R_azimuthal_angle = np.array([
        [np.cos(azimuthal_angle_rad), 0, np.sin(azimuthal_angle_rad)],
        [0, 1, 0],
        [-np.sin(azimuthal_angle_rad), 0, np.cos(azimuthal_angle_rad)]
    ]) # R_φ
    R_obliquity = np.array([
        [1, 0, 0],
        [0, np.cos(inclination_rad), np.sin(inclination_rad)],
        [0, -np.sin(inclination_rad), np.cos(inclination_rad)]
    ]) # R_θ
    R_arg_peri = np.array([
        [np.cos(arg_peri_rad), -np.sin(arg_peri_rad), 0],
        [np.sin(arg_peri_rad), np.cos(arg_peri_rad), 0],
        [0, 0, 1] # R_ψ
    ])

    R = R_azimuthal_angle @ R_obliquity @ R_arg_peri # R

    ring_normal_vector = R[:, 2] # vector RNA
    los_vector = np.array([0, 0, 1]) # vector LOS
    cos_angle = np.dot(ring_normal_vector, los_vector) # cosη
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    L = np.linspace(-size / 2, size / 2, size) # L
    X, Y = np.meshgrid(L, L) # X, Y

    m11, m12, m21, m22 = R[0, 0], R[0, 1], R[1, 0], R[1, 1]
    det_N = m11 * m22 - m12 * m21 # N - 2x2 R submatrix in the upper left corner

    if np.isclose(det_N, 0):
        return np.zeros((size, size), dtype=float)

    delta_x = (m22 * X - m12 * Y) / det_N # x_r
    delta_y = (-m21 * X + m11 * Y) / det_N # y_r

    radius_proj = np.sqrt(delta_x ** 2 + delta_y ** 2) # χ_r
    true_anomaly_val = np.arctan2(delta_y, delta_x) # ν_ring

    denominator = 1 + e * np.cos(true_anomaly_val) # 1 + ecosν_ring
    denominator[denominator <= 1e-9] = np.inf

    radius_inner_theory = (a_inner * (1 - e ** 2)) / denominator # χ_inner
    radius_outer_theory = (a_outer * (1 - e ** 2)) / denominator # χ_outer

    ring_mask = (radius_proj >= radius_inner_theory) & (radius_proj <= radius_outer_theory)

    transmission_map = np.zeros((size, size), dtype=float)
    if constant_fill:
        transmission_map[ring_mask] = fill
    else:
        transmission_map[ring_mask] = fill * abs(1 / cos_angle)

    return transmission_map

def quadratic_star_model(shape: list[int], coefficients: list[float]) -> np.array:
    """
    See formulas 2.2.27 - 2.2.30.
    Creates a star model using the quadratic limb darkening approximation
    :param shape: n, k
    :param coefficients: γ_1, γ_2
    :return:
    """

    n, k = shape
    n, k = int(n), int(k)

    u1, u2 = coefficients

    if n == 0 or k == 0: return np.zeros((1, 1))
    if n % 2 == 0: n -= 1
    if k % 2 == 0: k -= 1

    # Calculate the center of the matrix
    center_x = (n - 1) / 2
    center_y = (k - 1) / 2

    y, x = np.ogrid[:n, :k]

    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    star_radius_px = n / 2

    if star_radius_px == 0: star_radius_px = 1

    mu = np.cos(distance_from_center / star_radius_px * (math.pi / 2))

    result = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
    result[distance_from_center > star_radius_px] = 0
    result[result < 0] = 0

    return np.rot90(result)

def square_root_star_model(shape: list[int], coefficients: list[float]) -> np.array:
    """
    See formulas 2.2.27 - 2.2.29, 2.2.31.
    Creates a star model using the square-root limb darkening approximation
    :param shape: n, k
    :param coefficients: γ_3, γ_4
    :return:
    """

    n, k = shape
    n, k = int(n), int(k)

    u3, u4 = coefficients

    if n == 0 or k == 0: return np.zeros((1, 1))
    if n % 2 == 0: n -= 1
    if k % 2 == 0: k -= 1

    center_x = (n - 1) / 2
    center_y = (k - 1) / 2

    y, x = np.ogrid[:n, :k]

    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    star_radius_px = (n / 2)
    if star_radius_px == 0: star_radius_px = 1

    mu = np.cos(distance_from_center / star_radius_px * math.pi)

    result = 1 - u3 * (1 - mu) - u4 * (1 - mu) ** 2
    result[result < 0] = 0
    result[distance_from_center > star_radius_px] = 0

    return np.rot90(result)

def star_spot(star: np.array, angular_velocity=0., initial_longitude=0., time_change=0., radius=0., brightness=1.):
    """
    Add a star spot or a facula to the star model with proper coordinate mapping.

    Features are defined with:
    - longitude: 0 to 360 degrees (around the star)
    - radius: in units of stellar radii
    - brightness:
      * < 1: Dark spot (0 = completely dark, 1 = same as photosphere)
      * = 1: No effect (same as photosphere)
      * > 1: Bright facula (brighter than photosphere)
    """
    if not (radius == 0 or brightness == 1):
        size = star.shape[0]
        center = size // 2
        longitude = initial_longitude + angular_velocity * time_change
        if 180 <= longitude % 360 < 360:
            # Spot is on the far side of the star, return unchanged star
            return star
        lon = np.radians(longitude)  # Convert to radians
        model = star.copy()

        # Calculate spot size in pixels (convert from stellar radii to pixels)
        spot_radius_px = int(radius * center)
        if spot_radius_px < 1:
            spot_radius_px = 1  # Ensure at least 1 pixel

        x = np.cos(lon)
        spot_center_x = int(center + center * x)
        spot_center_y = center  # Equatorial position (y=0 in stellar coordinates)

        y_indices, x_indices = np.ogrid[:size, :size]

        # Calculate distance from spot center
        distance = np.sqrt((x_indices - spot_center_x) ** 2 + (y_indices - spot_center_y) ** 2)

        # Apply perspective distortion: spots appear squished horizontally near the edges
        # Calculate the foreshortening factor based on longitude
        foreshortening = abs(np.sin(lon))  # Maximum at 90°, zero at 0° and 180°
        
        # Adjust spot shape: compress horizontally based on viewing angle
        if foreshortening > 0.01:  # Avoid division by very small numbers
            # Create elliptical spot mask
            x_distance = np.abs(x_indices - spot_center_x)
            y_distance = np.abs(y_indices - spot_center_y)
            
            # Horizontal compression factor (1.0 = no compression, 0.1 = maximum compression)
            h_compression = foreshortening
            
            # Elliptical distance calculation
            elliptical_distance = np.sqrt((x_distance / max(h_compression, 0.1)) ** 2 + y_distance ** 2)
            
            feather = 1.
            spot_mask = np.clip(1 - (elliptical_distance - spot_radius_px + feather) / (2 * feather), 0, 1)
        else:
            # At extreme edges (longitude 0° or 180°), spot is very foreshortened
            feather = 1.
            spot_mask = np.clip(1 - (distance - spot_radius_px + feather) / (2 * feather), 0, 1)

        # Apply the spot/facula effect only where it's visible (on the stellar disk)
        on_disk = distance <= center

        if brightness < 1:
            spot_effect = 1.0 - (1.0 - brightness) * spot_mask
            model = np.where(on_disk, star * spot_effect, star)
        elif brightness > 1:
            facula_effect = 1.0 + (brightness - 1.0) * spot_mask
            model = np.where(on_disk, np.minimum(star * facula_effect, 2.0), star)

        model = np.clip(model, 0, 2)
        return model
    else:
        return star

def transit(star: np.array, mask: np.array, period: float, eccentricity: float, sma: float, inclination: float,
            longitude_of_ascending_node: float, argument_of_periapsis: float,
            steps: int = 500, angular_velocity=0, spot_longitude=0, spot_radius=0, spot_brightness=0) -> tuple:
    """
    See formulas 2.1.13 - 2.1.16 and 2.2.33 - 2.2.56
    Models the transit light curve based on orbital mechanics with optimization.

    :param star: The 2D array of the star.
    :param mask: The 2D transmission mask of the exoplanet with rings.
    :param period: P
    :param eccentricity: e_p
    :param sma: α_p
    :param inclination: i
    :param longitude_of_ascending_node: Ω
    :param argument_of_periapsis: ω
    :param steps: s
    :param angular_velocity: ω_spot
    :param spot_longitude: λ_0
    :param spot_radius: ρ
    :param spot_brightness: β
    :return: Δm(t)
    """
    print('Calculating the transit...')
    initial_intensity = np.sum(star)
    if initial_intensity == 0: return [], []
    data_points = []

    standard_inclination_rad = np.deg2rad(90.0 - inclination)
    lan_rad = np.deg2rad(longitude_of_ascending_node)
    arg_peri_rad = np.deg2rad(argument_of_periapsis)

    R_omega = np.array(
        [[np.cos(arg_peri_rad), -np.sin(arg_peri_rad), 0], [np.sin(arg_peri_rad), np.cos(arg_peri_rad), 0], [0, 0, 1]])
    R_i = np.array([[1, 0, 0], [0, np.cos(standard_inclination_rad), np.sin(standard_inclination_rad)],
                    [0, -np.sin(standard_inclination_rad), np.cos(standard_inclination_rad)]])
    R_Omega = np.array([[np.cos(lan_rad), 0, np.sin(lan_rad)], [0, 1, 0], [-np.sin(lan_rad), 0, np.cos(lan_rad)]])
    R_orbit = R_Omega @ R_i @ R_omega

    h_star, w_star = star.shape
    h_mask, w_mask = mask.shape
    center_x_star, center_y_star = w_star // 2, h_star // 2

    # finding the transit window
    star_radius_px = w_star / 2
    mask_radius_px = np.sqrt(h_mask ** 2 + w_mask ** 2) / 2
    max_dist_for_transit = star_radius_px + mask_radius_px

    # Convert orbital angles to radians for calculation.
    inclination_astro_rad = np.deg2rad(inclination)

    numerator = 1.0
    denominator = np.cos(inclination_astro_rad) * np.tan(lan_rad)

    # Handle edge cases where tan(lan_rad) is zero or infinite
    if abs(np.cos(lan_rad)) < 1e-9:  # Ω is 90 or 270 deg
        u_rad = np.pi / 2.0
    elif abs(np.sin(lan_rad)) < 1e-9:  # Ω is 0 or 180 deg
        u_rad = 0.0
    else:
        u_rad = np.arctan2(numerator, denominator)

    # There are two solutions for u (u and u+180°). We must choose the one when the planet is in front of the star
    z_sign_check = -(np.cos(u_rad) * np.sin(lan_rad) +
                     np.sin(u_rad) * np.cos(inclination_astro_rad) * np.cos(lan_rad))
    if z_sign_check < 0:
        u_rad += np.pi

    # Calculate the true anomaly (ν) from the argument of latitude (u)
    nu_center_rad = u_rad - arg_peri_rad

    # Convert true anomaly (ν) to eccentric anomaly (E).
    e = eccentricity
    E_center = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(nu_center_rad), e + np.cos(nu_center_rad))

    # Convert eccentric anomaly (E) to mean anomaly (mm) and then to time (t).
    M_center = E_center - e * np.sin(E_center)
    t_center = M_center * period / (2 * np.pi)

    if sma > 0:
        estimated_duration = (period * (star_radius_px + mask_radius_px)) / (np.pi * sma)
        half_search_window = 2.0 * estimated_duration
    else:  # Fallback for sma=0 case
        half_search_window = period / 100

    coarse_time_points = np.linspace(t_center - half_search_window, t_center + half_search_window, 400)

    t_start, t_end = None, None
    in_transit = False
    for t in coarse_time_points:
        M_A = mean_anomaly(t, period)
        E = eccentric_anomaly(M_A, eccentricity)
        nu = true_anomaly(E, eccentricity)
        r = radius_vector(sma, eccentricity, nu)
        pos_in_orbit = np.array([r * np.cos(np.deg2rad(nu)), r * np.sin(np.deg2rad(nu)), 0])
        pos_in_3d = R_orbit @ pos_in_orbit
        is_transiting = pos_in_3d[2] >= 0 and np.sqrt(pos_in_3d[0] ** 2 + pos_in_3d[1] ** 2) < max_dist_for_transit
        if is_transiting and not in_transit:
            t_start = t
            in_transit = True
        if in_transit and is_transiting:
            t_end = t
        if in_transit and not is_transiting:
            break

    if t_start is None:
        print('Unable to detect transit.')
        return [0, [(0, 0)]]

    transit_duration = t_end - t_start

    if transit_duration <= 0: return [0, [(0, 0)]]

    # Find the approximate time of minimum brightness
    padding = transit_duration * 0.1
    first_pass_steps = max(steps // 5, 100)
    first_pass_time_points = np.linspace(t_start - padding, t_end + padding, first_pass_steps)

    t_min = (t_start + t_end) / 2  # Fallback initialization
    min_intensity = float('inf')

    # Recalculate light curve centered around t_min
    total_duration = transit_duration + 2 * padding
    new_start_time = t_min - total_duration / 2
    new_end_time = t_min + total_duration / 2
    final_time_points = np.linspace(new_start_time, new_end_time, steps)

    for t in first_pass_time_points:
        M_A = mean_anomaly(t, period)
        E = eccentric_anomaly(M_A, eccentricity)
        nu = true_anomaly(E, eccentricity)
        r = radius_vector(sma, eccentricity, nu)
        pos_in_orbit = np.array([r * np.cos(np.deg2rad(nu)), r * np.sin(np.deg2rad(nu)), 0])
        pos_in_3d = R_orbit @ pos_in_orbit
        px, py, pz = pos_in_3d[0], pos_in_3d[1], pos_in_3d[2]

        light_blocked = 0.0
        if pz >= 0:
            tl_x, tl_y = int(round(center_x_star + px - w_mask / 2)), int(round(center_y_star + py - h_mask / 2))
            star_y_start, star_y_end = max(0, tl_y), min(h_star, tl_y + h_mask)
            star_x_start, star_x_end = max(0, tl_x), min(w_star, tl_x + w_mask)
            mask_y_start, mask_y_end = max(0, -tl_y), h_mask - max(0, (tl_y + h_mask) - h_star)
            mask_x_start, mask_x_end = max(0, -tl_x), w_mask - max(0, (tl_x + w_mask) - w_star)

            if star_y_end > star_y_start and star_x_end > star_x_start:
                star_slice = star_spot(star, angular_velocity, spot_longitude, t - new_start_time, spot_radius, spot_brightness)[star_y_start:star_y_end, star_x_start:star_x_end]
                mask_slice = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                if star_slice.shape == mask_slice.shape:
                    light_blocked = np.sum(star_slice - star_slice * np.exp(-mask_slice))

        current_intensity = initial_intensity - light_blocked
        if current_intensity < min_intensity:
            min_intensity = current_intensity
            t_min = t

    for t in final_time_points:
        M_A = mean_anomaly(t, period)
        E = eccentric_anomaly(M_A, eccentricity)
        nu = true_anomaly(E, eccentricity)
        r = radius_vector(sma, eccentricity, nu)
        pos_in_orbit = np.array([r * np.cos(np.deg2rad(nu)), r * np.sin(np.deg2rad(nu)), 0])
        pos_in_3d = R_orbit @ pos_in_orbit
        px, py, pz = pos_in_3d[0], pos_in_3d[1], pos_in_3d[2]

        light_blocked = 0.0
        if pz >= 0:
            tl_x, tl_y = int(round(center_x_star + px - w_mask / 2)), int(round(center_y_star + py - h_mask / 2))
            star_y_start, star_y_end = max(0, tl_y), min(h_star, tl_y + h_mask)
            star_x_start, star_x_end = max(0, tl_x), min(w_star, tl_x + w_mask)
            mask_y_start, mask_y_end = max(0, -tl_y), h_mask - max(0, (tl_y + h_mask) - h_star)
            mask_x_start, mask_x_end = max(0, -tl_x), w_mask - max(0, (tl_x + w_mask) - w_star)

            if star_y_end > star_y_start and star_x_end > star_x_start:
                star_slice = star_spot(star, angular_velocity, spot_longitude, t - new_start_time, spot_radius, spot_brightness)[star_y_start:star_y_end, star_x_start:star_x_end]
                mask_slice = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                if star_slice.shape == mask_slice.shape:
                    light_blocked = np.sum(star_slice - star_slice * np.exp(-mask_slice))

        current_intensity = initial_intensity - light_blocked

        # Normalize time to phase [0, 1] based on the new centered time window
        phase = (t - new_start_time) / total_duration
        data_points.append((phase, current_intensity))

    return total_duration, light_curve(data_points)

def correct(array: np.array) -> np.array:
    """
    See formula 2.2.29.
    Corrects the given array to the range [0, 1]

    :param np.array array: the array to normalize
    :return: normalized array
    """
    _max = 0.

    for x in range(len(array)):
        for y in range(len(array[0])):
            if array[x][y] > _max:
                _max = array[x][y]


    for x in range(len(array)):
        for y in range(len(array[0])):
            array[x][y] = array[x][y]/max
    return array

def show_model(model: np.ndarray, title='Optical Depth Distribution Map', x_label='x', y_label='y', colorbar_label='Optical Depth', vmin=None, vmax=None) -> None:
    """
    Shows the given model as an image using matplotlib

    :param model: The 2D numpy array to visualize
    :param title: Title of the plot
    :param x_label: Label for x-axis
    :param y_label: Label for y-axis
    :param colorbar_label: Label for the colorbar
    :param vmin: Minimum data value that corresponds to colormap min (default: None for auto)
    :param vmax: Maximum data value that corresponds to colormap max (default: None for auto)
    """
    plt.figure(figsize=(8, 8))

    # Make a copy to avoid modifying the original
    model = np.array(model, copy=True)

    # If the model is all zeros, skip plotting
    if np.all(model == 0):
        print("Warning: Model contains only zeros")
        plt.close()
        return

    # Normalize the model to [0, 1] range for better visualization
    model_min = np.min(model)
    model_max = np.max(model)
    if model_max > model_min:  # Avoid division by zero
        model = (model - model_min) / (model_max - model_min)

    # Set default vmin and vmax if not provided
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = 1.0

    # Use a perceptually uniform colormap that works well for stars
    cmap = 'viridis'  # or 'plasma', 'inferno', 'magma', 'cividis'

    # Create the plot with aspect='equal' to prevent distortion
    img = plt.imshow(model, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    # Add a colorbar
    cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    # Add title and labels
    plt.title(title, pad=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Remove axis ticks for a cleaner look
    plt.xticks([])
    plt.yticks([])

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()

def transit_animation(star: np.array, mask: np.array, period: float, eccentricity: float, sma: float, inclination: float, longitude_of_ascending_node: float, argument_of_periapsis: float,
            steps: int = 500, angular_velocity=0, spot_longitude=0, spot_radius=0, spot_brightness=0) -> list:
    """
    See formulas 2.1.13 - 2.1.16 and 2.2.33 - 2.2.56
    Models the transit animation based on orbital mechanics.
    Returns frames for all time points that the transit function uses for the light curve.

    :param star: The 2D array of the star.
    :param mask: The 2D transmission mask of the exoplanet with rings.
    :param period: P
    :param eccentricity: e_p
    :param sma: α_p
    :param inclination: i
    :param longitude_of_ascending_node: Ω
    :param argument_of_periapsis: ω
    :param steps: s
    :param angular_velocity: ω_spot
    :param spot_longitude: λ_0
    :param spot_radius: ρ
    :param spot_brightness: β
    :return: List of QImage frames for each time point in the transit
    """
    frames = []
    initial_intensity = np.sum(star)
    if initial_intensity == 0: return []

    standard_inclination_rad = np.deg2rad(90.0 - inclination)
    lan_rad = np.deg2rad(longitude_of_ascending_node)
    arg_peri_rad = np.deg2rad(argument_of_periapsis)

    R_omega = np.array(
        [[np.cos(arg_peri_rad), -np.sin(arg_peri_rad), 0], [np.sin(arg_peri_rad), np.cos(arg_peri_rad), 0], [0, 0, 1]])
    R_i = np.array([[1, 0, 0], [0, np.cos(standard_inclination_rad), np.sin(standard_inclination_rad)],
                    [0, -np.sin(standard_inclination_rad), np.cos(standard_inclination_rad)]])
    R_Omega = np.array([[np.cos(lan_rad), 0, np.sin(lan_rad)], [0, 1, 0], [-np.sin(lan_rad), 0, np.cos(lan_rad)]])
    R_orbit = R_Omega @ R_i @ R_omega

    h_star, w_star = star.shape
    h_mask, w_mask = mask.shape
    center_x_star, center_y_star = w_star // 2, h_star // 2

    # finding the transit window
    star_radius_px = w_star / 2
    mask_radius_px = np.sqrt(h_mask ** 2 + w_mask ** 2) / 2
    max_dist_for_transit = star_radius_px + mask_radius_px

    # Convert orbital angles to radians for calculation.
    inclination_astro_rad = np.deg2rad(inclination)

    numerator = 1.0
    denominator = np.cos(inclination_astro_rad) * np.tan(lan_rad)

    # Handle edge cases where tan(lan_rad) is zero or infinite
    if abs(np.cos(lan_rad)) < 1e-9:  # Ω is 90 or 270 deg
        u_rad = np.pi / 2.0
    elif abs(np.sin(lan_rad)) < 1e-9:  # Ω is 0 or 180 deg
        u_rad = 0.0
    else:
        u_rad = np.arctan2(numerator, denominator)

    # There are two solutions for u (u and u+180°). We must choose the one when the planet is in front of the star
    z_sign_check = -(np.cos(u_rad) * np.sin(lan_rad) +
                     np.sin(u_rad) * np.cos(inclination_astro_rad) * np.cos(lan_rad))
    if z_sign_check < 0:
        u_rad += np.pi

    # Calculate the true anomaly (ν) from the argument of latitude (u)
    nu_center_rad = u_rad - arg_peri_rad

    # Convert true anomaly (ν) to eccentric anomaly (E).
    e = eccentricity
    E_center = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(nu_center_rad), e + np.cos(nu_center_rad))

    # Convert eccentric anomaly (E) to mean anomaly (mm) and then to time (t).
    M_center = E_center - e * np.sin(E_center)
    t_center = M_center * period / (2 * np.pi)

    if sma > 0:
        estimated_duration = (period * (star_radius_px + mask_radius_px)) / (np.pi * sma)
        half_search_window = 2.0 * estimated_duration
    else:  # Fallback for sma=0 case
        half_search_window = period / 100

    coarse_time_points = np.linspace(t_center - half_search_window, t_center + half_search_window, 400)

    t_start, t_end = None, None
    in_transit = False
    for t in coarse_time_points:
        M_A = mean_anomaly(t, period)
        E = eccentric_anomaly(M_A, eccentricity)
        nu = true_anomaly(E, eccentricity)
        r = radius_vector(sma, eccentricity, nu)
        pos_in_orbit = np.array([r * np.cos(np.deg2rad(nu)), r * np.sin(np.deg2rad(nu)), 0])
        pos_in_3d = R_orbit @ pos_in_orbit
        is_transiting = pos_in_3d[2] >= 0 and np.sqrt(pos_in_3d[0] ** 2 + pos_in_3d[1] ** 2) < max_dist_for_transit
        if is_transiting and not in_transit:
            t_start = t
            in_transit = True
        if in_transit and is_transiting:
            t_end = t
        if in_transit and not is_transiting:
            break

    if t_start is None:
        print('Unable to detect transit.')
        return []

    transit_duration = t_end - t_start

    if transit_duration <= 0: return []

    # Find the approximate time of minimum brightness
    padding = transit_duration * 0.1
    first_pass_steps = max(steps // 5, 100)
    first_pass_time_points = np.linspace(t_start - padding, t_end + padding, first_pass_steps)

    t_min = (t_start + t_end) / 2  # Fallback initialization
    min_intensity = float('inf')

    # Calculate new_start_time before first pass (same as transit function)
    total_duration = transit_duration + 2 * padding
    new_start_time = t_min - total_duration / 2

    for t in first_pass_time_points:
        M_A = mean_anomaly(t, period)
        E = eccentric_anomaly(M_A, eccentricity)
        nu = true_anomaly(E, eccentricity)
        r = radius_vector(sma, eccentricity, nu)
        pos_in_orbit = np.array([r * np.cos(np.deg2rad(nu)), r * np.sin(np.deg2rad(nu)), 0])
        pos_in_3d = R_orbit @ pos_in_orbit
        px, py, pz = pos_in_3d[0], pos_in_3d[1], pos_in_3d[2]

        light_blocked = 0.0
        if pz >= 0:
            tl_x, tl_y = int(round(center_x_star + px - w_mask / 2)), int(round(center_y_star + py - h_mask / 2))
            star_y_start, star_y_end = max(0, tl_y), min(h_star, tl_y + h_mask)
            star_x_start, star_x_end = max(0, tl_x), min(w_star, tl_x + w_mask)
            mask_y_start, mask_y_end = max(0, -tl_y), h_mask - max(0, (tl_y + h_mask) - h_star)
            mask_x_start, mask_x_end = max(0, -tl_x), w_mask - max(0, (tl_x + w_mask) - w_star)

            if star_y_end > star_y_start and star_x_end > star_x_start:
                star_slice = star_spot(star, angular_velocity, spot_longitude, t - new_start_time, spot_radius, spot_brightness)[star_y_start:star_y_end, star_x_start:star_x_end]
                mask_slice = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                if star_slice.shape == mask_slice.shape:
                    light_blocked = np.sum(star_slice - star_slice * np.exp(-mask_slice))

        current_intensity = initial_intensity - light_blocked
        if current_intensity < min_intensity:
            min_intensity = current_intensity
            t_min = t

    # Recalculate time points centered around t_min (same as transit function)
    new_start_time = t_min - total_duration / 2
    new_end_time = t_min + total_duration / 2
    final_time_points = np.linspace(new_start_time, new_end_time, steps)

    # Generate frames for all final_time_points (same points used in transit function)
    for t in final_time_points:
        M_A = mean_anomaly(t, period)
        E = eccentric_anomaly(M_A, eccentricity)
        nu = true_anomaly(E, eccentricity)
        r = radius_vector(sma, eccentricity, nu)
        pos_in_orbit = np.array([r * np.cos(np.deg2rad(nu)), r * np.sin(np.deg2rad(nu)), 0])
        pos_in_3d = R_orbit @ pos_in_orbit
        px, py, pz = pos_in_3d[0], pos_in_3d[1], pos_in_3d[2]

        current_frame = star_spot(star, angular_velocity, spot_longitude, t - new_start_time, spot_radius, spot_brightness).copy()
        if pz >= 0:
            tl_x, tl_y = int(round(center_x_star + px - w_mask / 2)), int(round(center_y_star + py - h_mask / 2))
            star_y_start, star_y_end = max(0, tl_y), min(h_star, tl_y + h_mask)
            star_x_start, star_x_end = max(0, tl_x), min(w_star, tl_x + w_mask)
            mask_y_start, mask_y_end = max(0, -tl_y), h_mask - max(0, (tl_y + h_mask) - h_star)
            mask_x_start, mask_x_end = max(0, -tl_x), w_mask - max(0, (tl_x + w_mask) - w_star)

            if star_y_end > star_y_start and star_x_end > star_x_start:
                star_slice = current_frame[star_y_start:star_y_end, star_x_start:star_x_end]
                mask_slice = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                if star_slice.shape == mask_slice.shape:
                    # Apply opacity mask by subtracting the blocked light
                    star_slice *= np.exp(-mask_slice)

        frames.append(_array_to_qimage(current_frame))

    return frames


def _array_to_qimage(array: np.array) -> QImage:
    """
    Converts a 2D numpy array to a QImage for animation display.

    :param np.array array: array to convert
    :return: QImage representing the array as a grayscale image.
    """
    # Normalize array to 0-255
    normalized = (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-8) * 255
    img_array = normalized.astype(np.uint8)
    height, width = img_array.shape

    # Create QImage with a copy to avoid segmentation fault
    return QImage(img_array.data, width, height, width, QImage.Format.Format_Grayscale8).copy()

if __name__ == '__main__':
    for l in range(0, 360, 30):
        show_model(star_spot(quadratic_star_model([1000, 1000], [0.9, 0.9]), angular_velocity=0, initial_longitude=l, time_change=0, radius=0.1, brightness=2))
