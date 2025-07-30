import math

from PyQt6.QtGui import QImage
import matplotlib.pyplot as plt
import numpy as np

from formulas import light_curve, mean_anomaly, eccentric_anomaly, true_anomaly, radius_vector

"""
Uses numpy arrays to model the event of the transit

0 - minimum matrix pixel brightness
1 - maximum matrix pixel brightness
"""

def disk(radius: float, size: float = None) -> np.ndarray:
    """
    See formulas 2.1.1, 2.1.2
    Draws a disk of specified radius.
    Notion! Here, infinity is represented with number 1.

    :param radius: χ_R
    :param size: χ_a
    :return: numpy array with a disk of specified radius with values of τ(x, y)
    """
    if size is None:
        size = 2 * radius + 1 # default sizing
    size = int(round(size))
    if size == 0:
        return np.ones((1, 1))
    if size % 2 == 0:
        size -= 1

    center = (size - 1) / 2.0

    ans = np.ones((size, size), dtype=float)
    y, x = np.ogrid[:size, :size]

    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    ans[mask] = 0
    return ans

# must be edited
def elliptical_ring(
        size: int,
        a: float,
        e: float,
        w: float,
        obliquity: float,
        azimuthal_angle: float,
        argument_of_periapsis: float,
        fill: float) -> np.array:
    """
    See equations 2.2.3 - 2.2.23.
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

    L = np.linspace(-size / 2, size / 2, size) # L
    X, Y = np.meshgrid(L, L) # X, Y

    m11, m12, m21, m22 = R[0, 0], R[0, 1], R[1, 0], R[1, 1]
    det_N = m11 * m22 - m12 * m21 # N - 2x2 R submatrix in the upper left corner

    if np.isclose(det_N, 0):
        return np.ones((size, size), dtype=float)

    delta_x = (m22 * X - m12 * Y) / det_N # x - x_c
    delta_y = (-m21 * X + m11 * Y) / det_N # y - y_c

    radius_proj = np.sqrt(delta_x ** 2 + delta_y ** 2) # χ_r
    true_anomaly_val = np.arctan2(delta_y, delta_x) # ν_ring

    denominator = 1 + e * np.cos(true_anomaly_val) # 1 + ecosν_ring
    denominator[denominator <= 1e-9] = np.inf

    radius_inner_theory = (a_inner * (1 - e ** 2)) / denominator # χ_inner
    radius_outer_theory = (a_outer * (1 - e ** 2)) / denominator # χ_outer

    ring_mask = (radius_proj >= radius_inner_theory) & (radius_proj <= radius_outer_theory)

    transmission_map = np.ones((size, size), dtype=float)
    transmission_map[ring_mask] = fill

    return transmission_map

def quadratic_star_model(shape: list[int], coefficients: list[float]) -> np.array:
    """
    See formulas 2.2.24 - 2.2.27.
    Creates a star model using the quadratic limb darkening approximation
    :param shape: n, k
    :param coefficients: γ_1, γ_2
    :return:
    """

    n, k = shape
    u1, u2 = coefficients

    if n == 0:
        return np.zeros((1, 1))
    if k == 0:
        return np.zeros((1, 1))

    if n % 2 == 0:
        n -= 1

    if k % 2 == 0:
        k -= 1

    # Create an empty matrix
    result = np.zeros((n, k))

    # Calculate the center of the matrix
    center_x = (n - 1) / 2
    center_y = (k - 1) / 2

    # Fill the matrix with normalized intensities
    for x in range(n):
        for y in range(k):
            mu = np.cos(np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / (n / 2) * math.pi / 2)
            if mu < 0:
                mu = 0
            result[x, y] = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
            if result[x, y] < 0:
                result[x, y] = 0

    return np.rot90(result)


def square_root_star_model(shape: list[int], coefficients: list[float]) -> np.array:
    """
    See formulas 2.2.24 - 2.2.26, 2.2.28.
    Creates a star model using the square-root limb darkening approximation
    :param shape: n, k
    :param coefficients: γ_3, γ_4
    :return:
    """

    n, k = shape
    u3, u4 = coefficients

    if n == 0:
        return np.zeros((1, 1))
    if k == 0:
        return np.zeros((1, 1))

    if n % 2 == 0:
        n -= 1

    if k % 2 == 0:
        k -= 1

    # Create an empty matrix
    result = np.zeros((n, k))

    # Calculate the center of the matrix
    center_x = (n - 1) / 2
    center_y = (k - 1) / 2

    # Fill the matrix with normalized intensities
    for x in range(n):
        for y in range(k):
            mu = np.cos(np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / n * math.pi)
            if mu < 0:
                mu = 0
            result[x, y] = 1 - u3 * (1 - mu) - u4 * (1 - mu) ** 2
            if result[x, y] < 0:
                result[x, y] = 0

    return np.rot90(result)

# must be edited or deleted
def crop(array: np.array, rows: int, shape: tuple[int] = None, end: bool = False) -> np.array:
    """
    Crops the given array to the specified number of rows. Used for modeling the transit stages.

    :param np.array array: numpy array to be cropped
    :param int rows: number of rows to crop
    :param tuple shape: shape of the cropped array (optional, default is the original shape)
    :param bool end: if True, the cropped array will start from the end
    :return: cropped array
    """

    if not shape:
        shape = array.shape

    if array.size == 0:
        return np.array([])
    if rows > shape[0]:
        return np.zeros_like(shape)

    result = np.zeros_like(shape)
    if rows == 0:
        return array
    else:
        if array.shape[0] >= shape[0]:
            if end:
                    result[rows:] = array[:-rows]
            else:
                result[:-rows] = array[rows:]
        return result

# must be edited
def transit(star: np.array, mask: np.array, period: float, eccentricity: float, sma: float, inclination: float, longitude_of_ascending_node: float, argument_of_periapsis: float,
            steps: int = 500) -> list:
    """
    See formulas 2.1.13 - 2.1.16 and 2.2.30 - 2.2.53
    Models the transit light curve based on orbital mechanics.

    :param star: The 2D array of the star.
    :param mask: The 2D transmission mask of the exoplanet with rings.
    :param period: P
    :param eccentricity: e_p
    :param sma: α_p
    :param inclination: i
    :param longitude_of_ascending_node: Ω
    :param argument_of_periapsis: ω
    :param steps: s
    :return: Δm(t)
    """
    initial_intensity = np.sum(star) # I_0
    data_points = []

    # Define rotation matrices for the planet's orbit projection
    standard_inclination_rad = np.deg2rad(90.0 - inclination)
    lan_rad = np.deg2rad(longitude_of_ascending_node)
    arg_peri_rad = np.deg2rad(argument_of_periapsis)

    R_omega = np.array([
        [np.cos(arg_peri_rad), -np.sin(arg_peri_rad), 0],
        [np.sin(arg_peri_rad), np.cos(arg_peri_rad), 0],
        [0, 0, 1]
    ]) # R_ω
    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(standard_inclination_rad), np.sin(standard_inclination_rad)],
        [0, -np.sin(standard_inclination_rad), np.cos(standard_inclination_rad)]
    ]) # R_i
    R_Omega = np.array([
        [np.cos(lan_rad), 0, np.sin(lan_rad)],
        [0, 1, 0],
        [-np.sin(lan_rad), 0, np.cos(lan_rad)]
    ]) # R_Ω
    R_orbit = R_Omega @ R_i @ R_omega # R_o

    h_star, w_star = star.shape # h_S, w_S
    h_mask, w_mask = mask.shape # h, w
    center_x_star, center_y_star = w_star // 2, h_star // 2 # x_c, y_c

    # Iterate through time over one period, centered on the transit
    time_points = np.linspace(-period / 2, period / 2, steps) # t
    for t in time_points:
        # Calculate orbital position
        M_A = mean_anomaly(t, period)
        E = eccentric_anomaly(M_A, eccentricity)
        nu = true_anomaly(E, eccentricity)
        r = radius_vector(sma, eccentricity, nu)

        pos_in_orbit = np.array([r * np.cos(np.deg2rad(nu)), r * np.sin(np.deg2rad(nu)), 0]) # vector r

        pos_in_3d = R_orbit @ pos_in_orbit # vector p
        px, py, pz = pos_in_3d[0], pos_in_3d[1], pos_in_3d[2]

        current_intensity = initial_intensity

        # Check if the planet is in front of the star (pz > 0)
        if pz >= 0:
            current_star = star.copy()

            tl_x = int(round(center_x_star + px - w_mask / 2)) # x_t
            tl_y = int(round(center_y_star + py - h_mask / 2)) # y_t

            star_y_start = max(0, tl_y) # y_S_0
            star_y_end = min(h_star, tl_y + h_mask) # y_S'
            star_x_start = max(0, tl_x) # x_S_0
            star_x_end = min(w_star, tl_x + w_mask) # x_S'

            mask_y_start = max(0, -tl_y) # y_0
            mask_y_end = h_mask - max(0, (tl_y + h_mask) - h_star) # y'
            mask_x_start = max(0, -tl_x) # x_0
            mask_x_end = w_mask - max(0, (tl_x + w_mask) - w_star) # x'

            # Apply mask only if there is an overlap
            if star_y_end > star_y_start and star_x_end > star_x_start:
                star_slice = current_star[star_y_start:star_y_end, star_x_start:star_x_end]
                mask_slice = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

                if star_slice.shape == mask_slice.shape:
                    star_slice *= mask_slice

            current_intensity = np.sum(current_star)

        data_points.append((t, current_intensity)) # I(t)

    # Add out-of-transit points for a full light curve
    full_data = [(time_points[0] - period / steps, initial_intensity)] + data_points + [
        (time_points[-1] + period / steps, initial_intensity)] # I(t)

    return light_curve(full_data) # Δm(t)

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

def show_model(model: np.ndarray) -> None:
    """
    Shows the given model as an image using matplotlib

    :param np.ndarray model: the model to show
    """
    plt.figure()
    plt.imshow(model, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.show()

def transit_animation(star: np.array, mask: np.array, period: float, eccentricity: float, sma: float, inclination: float, longitude_of_ascending_node: float, argument_of_periapsis: float,
            steps: int = 500) -> list:
    """
    See formulas 2.1.13 - 2.1.16 and 2.2.30 - 2.2.53
    Models the transit animation based on orbital mechanics.

    :param star: The 2D array of the star.
    :param mask: The 2D transmission mask of the exoplanet with rings.
    :param period: P
    :param eccentricity: e_p
    :param sma: α_p
    :param inclination: i
    :param longitude_of_ascending_node: Ω
    :param argument_of_periapsis: ω
    :param steps: s
    :return: Δm(t)
    """
    frames = []

    # Define rotation matrices for the planet's orbit projection
    standard_inclination_rad = np.deg2rad(90.0 - inclination)
    azimuthal_tilt_rad = np.deg2rad(longitude_of_ascending_node)
    arg_peri_rad = np.deg2rad(argument_of_periapsis)

    R_omega = np.array([
        [np.cos(arg_peri_rad), -np.sin(arg_peri_rad), 0],
        [np.sin(arg_peri_rad), np.cos(arg_peri_rad), 0],
        [0, 0, 1]
    ])
    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(standard_inclination_rad), np.sin(standard_inclination_rad)],
        [0, -np.sin(standard_inclination_rad), np.cos(standard_inclination_rad)]
    ])
    R_Omega = np.array([
        [np.cos(azimuthal_tilt_rad), 0, np.sin(azimuthal_tilt_rad)],
        [0, 1, 0],
        [-np.sin(azimuthal_tilt_rad), 0, np.cos(azimuthal_tilt_rad)]
    ])
    R_orbit = R_Omega @ R_i @ R_omega

    h_star, w_star = star.shape
    h_mask, w_mask = mask.shape
    center_x_star, center_y_star = w_star // 2, h_star // 2

    # Iterate through time to generate frames
    time_points = np.linspace(-period / 2, period / 2, steps)
    for t in time_points:
        # Calculate orbital position
        M_A = mean_anomaly(t, period)
        E = eccentric_anomaly(M_A, eccentricity)
        nu = true_anomaly(E, eccentricity)
        r = radius_vector(sma, eccentricity, nu)

        # Position vector in the orbital plane
        pos_in_orbit = np.array([r * np.cos(np.deg2rad(nu)), r * np.sin(np.deg2rad(nu)), 0])

        # Project position onto the observer's 3D frame
        pos_in_3d = R_orbit @ pos_in_orbit
        px, py, pz = pos_in_3d[0], pos_in_3d[1], pos_in_3d[2]

        current_frame = star.copy()

        # Check if the planet is in front of the star (pz > 0)
        if pz >= 0:
            # Define top-left corner for placing the mask
            tl_x = int(round(center_x_star + px - w_mask / 2))
            tl_y = int(round(center_y_star + py - h_mask / 2))

            # Determine the overlapping region to avoid index errors
            star_y_start = max(0, tl_y)
            star_y_end = min(h_star, tl_y + h_mask)
            star_x_start = max(0, tl_x)
            star_x_end = min(w_star, tl_x + w_mask)

            mask_y_start = max(0, -tl_y)
            mask_y_end = h_mask - max(0, (tl_y + h_mask) - h_star)
            mask_x_start = max(0, -tl_x)
            mask_x_end = w_mask - max(0, (tl_x + w_mask) - w_star)

            # Apply mask only if there is a valid overlap
            if star_y_end > star_y_start and star_x_end > star_x_start:
                star_slice = current_frame[star_y_start:star_y_end, star_x_start:star_x_end]
                mask_slice = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

                # Ensure shapes match for multiplication
                if star_slice.shape == mask_slice.shape:
                    star_slice *= mask_slice

        # If planet is behind the star (pz < 0), current_frame remains the unmodified star
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
    show_model(disk(20, 200) + elliptical_ring(200, 40, 0.4, 1, 90, 50, 40, 0.4))

