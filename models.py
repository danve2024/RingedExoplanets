import math

from PyQt6.QtGui import QImage
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, sqrt, radians

from formulas import light_curve
from measure import Measure
from typing import Union

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
        size: Union[float, Measure.Unit],
        a: Union[float, Measure.Unit],
        e: Union[float, Measure.Unit],
        w: Union[float, Measure.Unit],
        obliquity: Union[float, Measure.Unit],
        azimuthal_angle: Union[float, Measure.Unit],
        argument_of_periapsis: Union[float, Measure.Unit],
        fill: float,
        focus: tuple = None) -> np.array:
    """
    Draws an elliptical ring using standard orbital parameters for orientation

    :param size: matrix size (used for matrix concatenation)
    :param a: ring semi-major axis (in pixel units)
    :param e: ring eccentricity
    :param w: ring width (in pixel units)
    :param obliquity: inclination angle (i) in degrees
    :param azimuthal_angle: longitude of ascending node (Ω) in degrees
    :param argument_of_periapsis: argument of periapsis (ω) in degrees
    :param fill: ring fill percentage (depends on the absorption coefficient)
    :param focus: focus coordinates of the ring (in pixel units)
    :return: numpy array with an elliptical ring of specified parameters
    """

    size = int(round(size))

    if size == 0:
        return np.ones((1, 1))

    if size % 2 == 0:
        size -= 1

    shape = (size, size)
    a = float(a)
    w = float(w)

    if focus is None:
        cxy = (size - 1) / 2.0
        focus = (cxy, cxy)
    fy, fx = focus
    c = e * a
    b0 = sqrt(a ** 2 - c ** 2)

    # Create grid
    y, x = np.ogrid[:size, :size]
    x_centered = x - fx
    y_centered = y - fy

    # Convert angles to radians
    i = radians(inclination)
    Ω = radians(omega)
    ω = radians(argument)

    # Apply rotations in the correct order:
    # 1. Rotate by argument of periapsis (ω) around z-axis
    # 2. Rotate by inclination (i) around x-axis
    # 3. Rotate by longitude of ascending node (Ω) around z-axis

    # First rotation (argument of periapsis)
    x_rot1 = x_centered * cos(ω) + y_centered * sin(ω)
    y_rot1 = -x_centered * sin(ω) + y_centered * cos(ω)

    # Second rotation (inclination)
    x_rot2 = x_rot1
    y_rot2 = y_rot1 * cos(i)
    # z component would be y_rot1 * sin(i), but we're projecting to 2D

    # Third rotation (longitude of ascending node)
    x_rot3 = x_rot2 * cos(Ω) + y_rot2 * sin(Ω)
    y_rot3 = -x_rot2 * sin(Ω) + y_rot2 * cos(Ω)

    # Apply the final rotated coordinates
    x_final = x_rot3 - c  # subtracting c to account for eccentricity
    y_final = y_rot3

    # Calculate the projected semi-minor axis
    b = b0 * sqrt(cos(i) ** 2 + (sin(i) * cos(Ω) ** 2))  # approximate projection

    if abs(b) < 1e-12:
        outer_mask = np.zeros(shape, dtype=bool)
        inner_mask = np.zeros(shape, dtype=bool)
    else:
        outer_mask = (x_final / a) ** 2 + (y_final / b) ** 2 <= 1
        a_inner = max(a - w, 1e-9)
        b_inner = max(b - w, 1e-9)
        inner_mask = (x_final / a_inner) ** 2 + (y_final / b_inner) ** 2 <= 1
    ring_mask = outer_mask & ~inner_mask

    arr = np.zeros(shape, dtype=float)
    arr[ring_mask] = fill - 1

    return arr

def quadratic_star_model(shape: list[int], coefficients: list[float]) -> np.array:
    """
    See formulas 2.2.3 - 2.2.6.
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
            mu = cos(np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / (n / 2) * math.pi / 2)
            if mu < 0:
                mu = 0
            result[x, y] = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
            if result[x, y] < 0:
                result[x, y] = 0

    return np.rot90(result)


def square_root_star_model(shape: list[int], coefficients: list[float]) -> np.array:
    """
    See formulas 2.2.3 - 2.2.5, 2.2.7.
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
            mu = cos(np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / n * math.pi)
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
def transit(star: np.array, mask: np.array, initial: float) -> list:
    """
        Models the transit by masking the star array with the array of the exoplanet with rings

        :param np.array star: the array of the star covered
        :param np.array mask: the array of the covering exoplanet with its rings
        :param float initial: the initial intensity of the star radiation
        :return: the list of data used for drawing a lightcurve
    """

    m, n = star.shape
    n_cover = mask.shape[0]

    data = []
    initial_intensity = initial  # calculating initial illuminance
    cutout_intensity = np.sum(star)

    if mask.shape[1] != n:
        raise ValueError(f"Mask array must have {n} columns to match the base array. But it has {mask.shape[1]}.")

    for start_row in range(m - 1, -n_cover, -1):
        result = star.copy()

        overlap_start = max(start_row, 0)
        overlap_end = min(start_row + n_cover, m)

        cover_start = overlap_start - start_row
        cover_end = cover_start + (overlap_end - overlap_start)

        # Apply the mask with max(0, star[x][y] - mask[x][y])
        result[overlap_start:overlap_end, :] = np.maximum(
            0, star[overlap_start:overlap_end, :] * mask[cover_start:cover_end, :]
        )

        # show_model(result)

        intensity = np.sum(result)  # calculating the sum of intensities from each pixel - Ii
        data.append(float(initial_intensity - cutout_intensity + intensity))  # adding Ii to the output array

    data.insert(0, initial_intensity)
    data.append(initial_intensity)

    return light_curve(data)  # [I1, I2, ..., Ii] -> [(Φ1, Δm1), (Φ2, Δm2), ..., (Φi, Δmi)]

def correct(array: np.array) -> np.array:
    """
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

def transit_animation(star: np.array, mask: np.array) -> list:
    """
        Models the transit by masking the star array with the array of the exoplanet with rings

        :param np.array star: the array of the star covered
        :param np.array mask: the array of the covering exoplanet with its rings
        :return: the list of data used for drawing a lightcurve
    """

    m, n = star.shape
    n_cover = mask.shape[0]

    frames = []

    if mask.shape[1] != n:
        raise ValueError(f"Mask array must have {n} columns to match the base array.")

    for start_row in range(m - 1, -n_cover, -1):
        result = star.copy()

        overlap_start = max(start_row, 0)
        overlap_end = min(start_row + n_cover, m)

        cover_start = overlap_start - start_row
        cover_end = cover_start + (overlap_end - overlap_start)

        # Apply the mask with max(0, star[x][y] - mask[x][y])
        result[overlap_start:overlap_end, :] = np.maximum(
            0, star[overlap_start:overlap_end, :] * mask[cover_start:cover_end, :]
        )

        frames.append(_array_to_qimage(result))  # adding a new frame

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

