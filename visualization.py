import os
import time
import sys
import traceback

from PyQt6.QtWidgets import QHBoxLayout, QSlider, QMessageBox, QFileDialog, QComboBox
from PyQt6.QtCore import QTimer, Qt, QBuffer, QIODevice
from PyQt6.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from units import *
from formulas import *
from observations import Observations

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout,
    QLabel, QLineEdit, QCheckBox, QPushButton, QDialog
)

from PIL import Image
from io import BytesIO

"""
Fixed:
- stellar radius
- stellar temperature
- log(g)
- ring specific absorption coefficient
- band/wavelength
- model pixel size

Sliders:
- exoplanet semi-major axis (fixed option)
- exoplanet orbit eccentricity (fixed option)
- exoplanet orbit inclination (fixed option)
- exoplanet longitude of ascending node (fixed option)
- exoplanet argument of periapsis (fixed option)
- exoplanet radius (fixed option)
- exoplanet mass (fixed option)
- ring density (fixed option) (dependent)
- ring eccentricity
- ring semi-major axis (dependent)
- ring width (dependent)
- ring obliquity
- ring azimuthal angle
- ring argument of periapsis
- ring mass (dependent)
"""

class Selection(QWidget):
    """
    A widget to select and configure parameters for the simulation.
    """
    def __init__(self, default_values):
        super().__init__()

        self.default_values = default_values
        self.user_values = {}

        self.checkboxes = {}
        self.mins = {}
        self.maxs = {}

        self.dependent_sliders = ['sma', 'width', 'mass', 'density']
        self.user_values['other'] = {}

        # Predefined values
        self.user_values['other']['sma_start'] = 'Exoplanet Radius'
        self.user_values['other']['sma_end'] = 'Hill Sphere'
        self.user_values['other']['ring_mass_start'] = 'Auto'
        self.user_values['other']['ring_mass_end'] = 'Avoid Binarity'
        self.user_values['other']['width_start'] = 0
        self.user_values['other']['width_end'] = 'Maximum Possible'
        self.user_values['other']['density_start'] = 0.01 * gcm3
        self.user_values['other']['density_end'] = 'Roche Critical Density'

        self.user_values['other']['specific_absorption_coefficient'] = 2.3e-3 * (m**2/g)
        self.user_values['other']['star_radius'] = 700_000 * km
        self.user_values['other']['star_temperature'] = 4000 * K
        self.user_values['other']['star_log_g'] = 2.00
        self.user_values['other']['wavelength'] = 3437 * angstrom
        self.user_values['other']['band'] = 'u'
        self.user_values['other']['limb_darkening'] = 'quadratic'
        self.user_values['other']['pixel_size'] = 10000 * km

        self.init_ui()

    def init_ui(self):
        layout = QGridLayout()  # Use a grid layout for column alignment

        # Set column headers
        layout.addWidget(QLabel("Parameter"), 0, 0)
        layout.addWidget(QLabel("Auto"), 0, 1)
        layout.addWidget(QLabel("Manual"), 0, 2)
        layout.addWidget(QLabel("Min"), 0, 3)
        layout.addWidget(QLabel("Max"), 0, 4)

        row = 1  # Start from the second row for parameters

        for param, measure in self.default_values.items():
            # Parameter Name Label
            if self.get_unit_label(param):
                param_label = QLabel(
                    f'{param.replace("_", " ").replace("sma", "semi-major axis")} ({self.get_unit_label(param)})')
            else:
                param_label = QLabel(param.replace("_", " ").replace("sma", "semi-major axis"))
            layout.addWidget(param_label, row, 0)

            if isinstance(measure, Measure):
                # Auto Checkbox (Default)
                auto_cb = QCheckBox("Auto")
                auto_cb.setChecked(True)  # Start with Auto checked
                auto_cb.stateChanged.connect(lambda state, p=param: self.auto(state, p))
                layout.addWidget(auto_cb, row, 1)

                # Manual Checkbox
                manual_cb = QCheckBox("Manual")
                manual_cb.stateChanged.connect(lambda state, p=param: self.manual(state, p))
                layout.addWidget(manual_cb, row, 2)
                if param == 'sma':
                    # Min and Max Dropdowns
                    min_input = QComboBox()
                    min_input.addItems(['Exoplanet Radius', 'Roche Limit'])
                    min_input.setEditable(True)
                    min_input.currentTextChanged.connect(self.set_sma_start)
                    max_input = QComboBox()
                    max_input.addItems(['Hill Sphere', 'Roche Limit'])
                    max_input.setEditable(True)
                    max_input.currentTextChanged.connect(self.set_sma_end)
                elif param == 'mass':
                    # Min and Max Dropdowns
                    min_input = QComboBox()
                    min_input.addItem('Auto')
                    min_input.setEditable(True)
                    min_input.currentTextChanged.connect(self.set_ring_mass_start)
                    max_input = QComboBox()
                    max_input.addItem('Avoid Binarity')
                    max_input.setEditable(True)
                    max_input.currentTextChanged.connect(self.set_ring_mass_end)
                elif param == 'width':
                    # Min Input and Max Dropdown
                    min_input = QLineEdit(str(10.))
                    max_input = QComboBox()
                    max_input.addItem('Maximum Possible')
                    max_input.setEditable(True)
                    max_input.currentTextChanged.connect(self.set_width_end)
                elif param == 'density':
                    # Min Input and Max Dropdown
                    min_input = QLineEdit(str(10.))
                    max_input = QComboBox()
                    max_input.addItem('Roche Critical Density')
                    max_input.setEditable(True)
                    max_input.currentTextChanged.connect(self.set_width_end)
                else:
                    # Min and Max Input Fields
                    min_input = QLineEdit(str(measure.min(measure.unit)))
                    max_input = QLineEdit(str(measure.max(measure.unit)))

                min_input.setEnabled(False)
                max_input.setEnabled(False)
                layout.addWidget(min_input, row, 3)
                layout.addWidget(max_input, row, 4)

                # Store references
                self.checkboxes[param] = (auto_cb, manual_cb)
                self.mins[param] = min_input
                self.maxs[param] = max_input

                row += 1  # Move to the next row for the next parameter

            else: # One-value parameters
                options = {
                    'specific_absorption_coefficient': ['0.0023'],
                    'star_radius': ['700000'],
                    'star_temperature': ['4000 (quadratic)', '4500 (quadratic)', '4600 (quadratic)', '4700 (quadratic)', '5300 (quadratic)', '5460 (quadratic)', '5500 (quadratic and square-root)', '5780 (quadratic)', '6000 (quadratic and square-root)', '6020 (quadraric)', '6300 (quadratic)', '6730 (quadratic)', '7000 (square-root)', '8000 (square-root)', '10000 (square-root)', '15000 (square-root)', '20000 (square-root)'],
                    'star_log(g)': ['2.00 (quadratic)', '2.50 (quadratic)', '3.00 (quadratic and square-root)', '3.44 (quadratic)', '3.50 (quadratic)', '4.00 (quadratic or square-root)', '4.44 (quadratic)', '4.50 (quadratic)', '4.60 (quadratic)'],
                    'wavelength': ['3437 (square-root)', '4212 (square-root)', '4687 (square-root)', '5475 (square-root)', '6975 (square-root)'],
                    'band': ['u (quadratic)', 'b (quadratic)', 'v (quadratic)', 'y (quadratic)', 'U (quadratic)', 'B (quadratic)', 'V (quadratic)'],
                    'limb_darkening': ['quadratic', 'square-root'],
                    'pixel_size': ['1000']
                }

                def set_specific_absorption_coefficient(value):
                    self.user_values['other']['specific_absorption_coefficient'] = float(value.split()[0]) * (m**2/g)

                def set_star_radius(value):
                    self.user_values['other']['angular_size'] = int(value.split()[0]) * km

                def set_star_temperature(value):
                    self.user_values['other']['star_temperature'] = int(value.split()[0])

                def set_star_log_g(value):
                    self.user_values['other']['star_log_g'] = float(value.split()[0])

                def set_wavelength(value):
                    self.user_values['other']['wavelength'] = int(value.split()[0])

                def set_band(value):
                    self.user_values['other']['band'] = value.split()[0]

                def set_limb_darkening(value):
                    self.user_values['other']['limb_darkening'] = value.split()[0]

                def set_pixel_size(value):
                    self.user_values['other']['pixel_size'] = int(value.split()[0]) * km

                actions = {
                    'specific_absorption_coefficient': set_specific_absorption_coefficient,
                    'star_radius': set_star_radius,
                    'star_temperature': set_star_temperature,
                    'star_log(g)': set_star_log_g,
                    'wavelength': set_wavelength,
                    'band': set_band,
                    'limb_darkening': set_limb_darkening,
                    'pixel_size': set_pixel_size
                }

                dropdown = QComboBox()
                dropdown.addItems(options[param])
                dropdown.currentTextChanged.connect(actions[param])
                if param in ['specific_absorption_coefficient', 'star_radius', 'pixel_size']:
                    dropdown.setEditable(True)
                layout.addWidget(dropdown, row, 1)

                row += 1

        # Done Button
        done_btn = QPushButton("Done")
        done_btn.clicked.connect(self.click)  # Connect to close function
        layout.addWidget(done_btn, row, 0, 1, 5)  # Span across all columns

        self.setLayout(layout)
        self.setWindowTitle("Parameter Selector")
        self.show()

    @staticmethod
    def get_unit_label(name: str) -> str:
        """Get the unit label for a given parameter"""
        units = {
            'specific_absorption_coefficient': 'm²/g',
            'exoplanet_sma': 'au',
            'exoplanet_orbit_eccentricity': '',
            'exoplanet_orbit_inclination': 'deg',
            'exoplanet_argument_of_periapsis': 'deg',
            'exoplanet_radius': 'km',
            'exoplanet_mass': 'kg',
            'density': 'g/cm³',
            'eccentricity': '',
            'sma': 'km',
            'width': 'km',
            'mass': 'kg',
            'obliquity': 'deg',
            'azimuthal_angle': 'deg',
            'argument_of_periapsis': 'deg',
            'star_radius': 'km',
            'star_temperature': 'K',
            'star_log(g)': '',
            'wavelength': 'Å',
            'pixel_size': 'km'
        }
        return units.get(name, '')

    def auto(self, state, param):
        """Auto select parameter"""
        if state == 2:  # Auto checked
            self.checkboxes[param][1].setChecked(False)  # Uncheck Manual
            self.mins[param].setEnabled(False)
            self.maxs[param].setEnabled(False)
        else:
            # Prevent unchecking both checkboxes
            if not self.checkboxes[param][1].isChecked():
                self.checkboxes[param][0].setChecked(True)  # Re-check Auto

    def manual(self, state, param):
        """Manually set the parameter values."""
        if state == 2:  # Manual checked
            self.checkboxes[param][0].setChecked(False)  # Uncheck Auto
            self.mins[param].setEnabled(True)
            self.maxs[param].setEnabled(True)
        else:
            # Prevent unchecking both checkboxes
            if not self.checkboxes[param][0].isChecked():
                self.checkboxes[param][1].setChecked(True)  # Re-check Manual

    def click(self):
        """Close the window and store user-defined parameter values."""
        for param in self.default_values.keys():
            if param in self.checkboxes:
                measure = self.default_values[param]

                if self.checkboxes[param][0].isChecked() or param in self.dependent_sliders:  # Auto selected or the slider is dependent
                    # Use existing measure values without changes
                    self.user_values[param] = Measure(measure.min(measure.unit), measure.max(measure.unit), measure.unit,
                                                      measure.label)

                    if param == 'width':
                        self.user_values['other']['width_start'] = float(self.mins[param].text())
                    elif param == 'density':
                        self.user_values['other']['density_start'] = float(self.mins[param].text())

                else:  # Manual selected
                    min_val = float(self.mins[param].text())
                    max_val = float(self.maxs[param].text())
                    # Create a new Measure object with user-defined min/max but keep unit and label from defaults
                    self.user_values[param] = Measure(min_val, max_val, measure.unit, measure.label)

        # Close the window after clicking Done
        self.close()

    def set_sma_start(self, dropdown: str):
        self.user_values['other']['sma_start'] = dropdown

    def set_sma_end(self, dropdown: str):
        self.user_values['other']['sma_end'] = dropdown

    def set_ring_mass_start(self, dropdown: str):
        self.user_values['other']['ring_mass_start'] = dropdown

    def set_ring_mass_end(self, dropdown: str):
        self.user_values['other']['ring_mass_end'] = dropdown

    def set_width_end(self, dropdown: str):
        self.user_values['other']['width_end'] = dropdown


class LoadFile(QWidget):
    """
    A widget to load an observations file for comparing observations with the model.
    """
    def __init__(self):
        super().__init__()
        self.button = None
        self.proceed = None
        self.filename = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Load observations file")
        self.setGeometry(500, 500, 400, 300)
        self.button = QPushButton("Open File", self)
        self.button.clicked.connect(self.openFileDialog)
        self.button.setGeometry(150, 130, 100, 30)

        self.proceed = QPushButton("No File", self)
        self.proceed.clicked.connect(self.close)
        self.proceed.setGeometry(150, 170, 100, 30)
        self.show()

    def openFileDialog(self):
        """
        A function for opening the file dialog and pre-processing the selected file.
        """
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("CSV Files (*.csv)") # Search for CSV files only
        file_dialog.setWindowTitle("Select File")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            self.filename = selected_files[0]
            self.close()


class Model(QWidget):
    """
    A widget to display the simulation model and control parameters sliders.
    """
    def __init__(self, parameters: dict, calculate_data, observations_file: str = None) -> None:
        from space import Star
        super().__init__()
        self.calculate_data = calculate_data

        if observations_file is None:
            self.observations = None
        else:
            self.observations = Observations(observations_file) # Working with the observation data if it is loaded
        self.exoplanet = None

        self.magnitude_shift = None
        self.magnitude_calibrating = None
        self.time_shift = None
        self.magnitude_shift_label = None
        self.time_shift_label = None
        self.magnitude_calibrating_label = None

        self.setWindowTitle("Ringed Exoplanet Transit Simulation")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.sliders = {}
        self.slider_labels = {}

        # Default parameters
        self.params = parameters.pop('other')
        self.sma_start = self.params['sma_start']
        self.sma_end = self.params['sma_end']
        self.density_start = self.params['density_start']
        self.density_end = self.params['density_end']
        self.ring_mass_start = self.params['ring_mass_start']
        self.ring_mass_end = self.params['ring_mass_end']
        self.width_start = self.params['width_start']
        self.width_end = self.params['width_end']
        self.specific_absorption_coefficient = self.params['specific_absorption_coefficient']
        self.temperature = self.params['star_temperature']
        self.log_g = self.params['star_log_g']
        self.star_radius = self.params['star_radius']
        self.wavelength = self.params['wavelength']
        self.band = self.params['band']
        self.limb_darkening = self.params['limb_darkening']
        self.pixel_size = self.params['pixel_size']
        self.defaults = parameters

        self.star = Star(self.star_radius, self.temperature, self.log_g, self.wavelength, self.band, self.pixel_size, self.limb_darkening)

        # Create sliders
        for key, measure in self.defaults.items():
            self.create_slider(key, measure)

        # Plot
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Time spent label
        self.time_label = QLabel("Time spent: 0.0 sec")
        self.layout.addWidget(self.time_label)

        # Transit duration label
        self.eclipse_label = QLabel("Transit duration: 0.0 sec")
        self.layout.addWidget(self.eclipse_label)

        # Sliders for working with the observed data
        if self.observations is not None:
            self.magnitude_shift_slider()
            self.magnitude_calibrating_slider()
            self.time_shift_slider()

        # Additional parameters
        data_label = QLabel(f"κ: {self.specific_absorption_coefficient / (m**2/g)}m²/g    R: {self.star_radius / km}km    T: {self.temperature / K}K    log(g): {self.log_g}    λ: {self.wavelength / angstrom}Å    band: {self.band}    px: {self.pixel_size / km}km")
        self.layout.addWidget(data_label, alignment=Qt.AlignmentFlag.AlignLeft)

        # Show animation button
        self.animation_button = QPushButton("Simulate Transit Animation")
        self.animation_button.clicked.connect(self.show_animation_window)
        self.layout.addWidget(self.animation_button, alignment=Qt.AlignmentFlag.AlignRight)

        for _ in range(3):
            self.update_plot()

    def create_slider(self, name, measure):
        """Create a slider"""
        container = QHBoxLayout()
        unit = self.get_unit_label(name)
        if unit:
            label = QLabel(f"{name.replace('_', ' ').replace('sma', 'semi-major axis').capitalize()} ({unit}):")
        else:
            label = QLabel(name.replace('_', ' ').replace('sma', 'semi-major axis').capitalize())
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(50)
        slider.valueChanged.connect(self.update_plot)
        value_label = QLabel(str(measure.min))

        container.addWidget(label)
        container.addWidget(slider)
        container.addWidget(value_label)

        self.layout.addLayout(container)
        self.sliders[name] = (slider, measure)
        self.slider_labels[name] = value_label

    def magnitude_shift_slider(self):
        """
        A slider for making major shifts (0-12 mag) of the magnitude of the by a specific value selected by the slider. It is needed to work with the magnitude change (produced by the model) instead of the magnitude itself (observed).
        """
        container = QHBoxLayout()
        label = QLabel('Observations magnitude shift')
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(-20)
        slider.setMaximum(20)
        slider.setValue(0)
        slider.valueChanged.connect(self.update_plot)
        value_label = QLabel('0')

        container.addWidget(label)
        container.addWidget(slider)
        container.addWidget(value_label)

        self.layout.addLayout(container)

        self.magnitude_shift = slider
        self.magnitude_shift_label = value_label

    def magnitude_calibrating_slider(self):
        """
            A slider for making minor shifts (0-1 mag) of the magnitude of the by a specific value selected by the slider. It is needed to work with the magnitude change (produced by the model) instead of the magnitude itself (observed).
        """
        container = QHBoxLayout()
        label = QLabel('Observations magnitude calibrating')
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(-1000)
        slider.setMaximum(1000)
        slider.setValue(0)
        slider.valueChanged.connect(self.update_plot)
        value_label = QLabel('0')

        container.addWidget(label)
        container.addWidget(slider)
        container.addWidget(value_label)

        self.layout.addLayout(container)

        self.magnitude_calibrating = slider
        self.magnitude_calibrating_label = value_label

    def time_shift_slider(self):
        """
        A slider for selecting the moment of the observations to compare with the model. The observation data is cropped by the x-axis and moved for the specific time span selected.
        """
        container = QHBoxLayout()
        label = QLabel('Observations phase shift')
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.valueChanged.connect(self.update_plot)
        value_label = QLabel('0')

        container.addWidget(label)
        container.addWidget(slider)
        container.addWidget(value_label)

        self.layout.addLayout(container)

        self.time_shift = slider
        self.time_shift_label = value_label

    @staticmethod
    def get_unit_label(name: str) -> str:
        """Get the unit label for a given parameter"""
        units = {
            'exoplanet_sma': 'au',
            'exoplanet_orbit_eccentricity': '',
            'exoplanet_orbit_inclination': 'deg',
            'exoplanet_argument_of_periapsis': 'deg',
            'exoplanet_radius': 'km',
            'exoplanet_mass': 'kg',
            'density': 'g/cm³',
            'eccentricity': '',
            'sma': 'km',
            'width': 'km',
            'mass': 'kg',
            'obliquity': 'deg',
            'azimuthal_angle': 'deg',
            'argument_of_periapsis': 'deg',
            'specific_absorption_coefficient': 'm²/g',
            'star_radius': 'km',
            'star_temperature': 'K',
            'star_log(g)': '',
            'wavelength': 'Å',
            'pixel_size': 'km'
        }
        return units.get(name, '')

    def update_plot(self) -> None:
        """Update the plot based on the current slider values"""
        start_time = time.time()

        params = {}
        for key, (slider, measure) in self.sliders.items():
            real_value = measure.slider(slider.value())
            params[key] = real_value
            self.slider_labels[key].setText(f"{real_value:.2f}")

        params['specific_absorption_coefficient'] = self.specific_absorption_coefficient
        params['star_temperature'] = self.temperature
        params['star_log_g'] = self.log_g
        params['star_radius'] = self.star_radius
        params['wavelength'] = self.wavelength
        params['band'] = self.band
        params['limb_darkening'] = self.limb_darkening
        params['pixel_size'] = self.pixel_size

        data, transit_duration, self.exoplanet = self.calculate_data(**params, star=self.star)

        self.update_dependent_sliders()

        if self.observations is not None: # shifting (magnitude to magnitude change and time to phase) and normalizing (time to phase) the observation data
            phase_shift = self.time_shift.value() / 100
            magnitude_shift = self.magnitude_shift.value()
            magnitude_calibrating = self.magnitude_calibrating.value() / 1000
            self.time_shift_label.setText(str(phase_shift))
            self.magnitude_shift_label.setText(str(magnitude_shift))
            self.magnitude_calibrating_label.setText(str(magnitude_calibrating))

            self.observations.shift(transit_duration, phase_shift, magnitude_shift + magnitude_calibrating)

        self.ax.clear()
        self.ax.set_title("Simulation Result")

        try:
            lightcurve = data
        except IndexError:
            lightcurve = [(0,0), (1,0)]

        # Plotting the observations data with the selected shifts
        if self.observations is None:
            x = [item[0] for item in lightcurve]
            y = [item[1] for item in lightcurve]
            self.ax.plot(x, y, label='model')
        else:
            x = [item[0] for item in data[1]]
            y = [item[1] for item in data[1]]
            self.ax.plot(x, y, label='model')
            if len(self.observations.data) > 1:
                xo, yo = zip(*self.observations.data)
                self.ax.plot(xo, yo, 'g', label='observations')
            else:
                pass

        self.ax.invert_yaxis()
        self.ax.set_xlabel("Phase")
        self.ax.set_ylabel("Magnitude Change")
        self.ax.legend()
        self.canvas.draw()

        elapsed_time = time.time() - start_time
        self.time_label.setText(f"Time spent: {elapsed_time:.2f} sec")
        self.eclipse_label.setText(f"Transit duration: {transit_duration/days:.2f} d")

    def update_dependent_sliders(self) -> None:
        """Update the dependent sliders (ring mass and semi-major axis)"""

        # Working with dropdowns

        if self.sma_start == 'Exoplanet Radius':
            a_min = self.exoplanet.radius
        elif self.sma_start == 'Roche Limit':
            a_min = self.exoplanet.min_roche_sma
        else:
            a_min = float(self.sma_start) * km

        if self.sma_end == 'Roche Limit':
            a_max = self.exoplanet.max_roche_sma
        elif self.sma_end == 'Hill Sphere':
            a_max = self.exoplanet.hill_sma(self.star.mass)
        else:
            a_max = float(self.sma_end) * km

        self.defaults['sma'].update(a_min / km, a_max / km)

        w_min = self.width_start * km
        if self.width_end == 'Maximum Possible':
            w_max = min(a_max - a_min, self.exoplanet.rings.sma / 2)
        else:
            w_max = float(self.width_end) * km

        self.defaults['width'].update(w_min / km, w_max / km)

        if self.ring_mass_end == 'Avoid Binarity':
            m_max = self.exoplanet.maximum_ring_mass
        else:
            m_max = float(self.ring_mass_end) * kg

        if self.ring_mass_start == 'Auto':
            m_min = 0.5 * m_max
        else:
            m_min = float(self.ring_mass_start) * kg

        self.defaults['mass'].update(m_min / kg, m_max / kg)

    def show_animation_window(self):
        """Open the animation window with error handling."""
        try:
            if hasattr(self, 'star') and hasattr(self, 'exoplanet'):
                animation_window = AnimationWindow(self.star, self.exoplanet, self)
                animation_window.exec()
            else:
                QMessageBox.warning(self, "Warning", "Please update the plot before running the animation.")
        except Exception as e:
            print("Error during animation:", e)
            traceback.print_exc()

class FramesWindow(QDialog):
    def __init__(self, frames, parent=None, title: str = "Ringed Exoplanet Transit Simulation", size: tuple[int] = (600, 600), modal: bool = True, suffix: str = ''):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(modal)
        self.resize(size[0], size[1])

        self.layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        self.frames = frames
        self.current_frame_index = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS

    def update_frame(self):
        """Update the animation frame safely."""
        if not self.frames:
            return  # Avoid update if frames are missing

        frame = self.frames[self.current_frame_index]
        if frame.isNull():
            return  # Skip invalid frames

        pixmap = QPixmap.fromImage(frame)
        self.label.setPixmap(pixmap)
        self.current_frame_index = (self.current_frame_index + 1) % len(self.frames)

    def save_frames(self, _dir="animation"):
        """
        Saves the stored QImage frames into a folder.

        :param _dir: Folder for saving the frames
        :return:
        """
        if _dir not in os.listdir():
            os.mkdir(_dir)
        for i in range(len(self.frames)):
            self.frames[i].save(f'{_dir}/{_dir}{i}.jpg')

    def save_gif(self, filename="animation.gif"):
        """
        Converts the stored QImage frames into a GIF animation and saves it.

        Args:
            filename (str): The name of the GIF file to save.
        """
        if not self.frames:
            print("No frames available to save as GIF.")
            return

        pil_images = []
        for i, qimage in enumerate(self.frames):
            if qimage.isNull():
                print(f"Skipping invalid frame at index {i}")
                continue

            # Convert QImage to bytes (PNG format) using a QBuffer
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            qimage.save(buffer, "PNG")
            buffer.close()

            # Load bytes into a PIL Image
            try:
                pil_image = Image.open(BytesIO(buffer.data()))
                pil_images.append(pil_image)
            except Exception as e:
                print(f"Error converting QImage to PIL Image for frame {i}: {e}")
                continue

        if not pil_images:
            print("No valid images could be converted for GIF creation.")
            return

        # Save the PIL images as an animated GIF
        try:
            # The first image is saved, and subsequent images are appended
            # duration is in milliseconds, matching the QTimer interval
            # loop=0 means the GIF will loop indefinitely
            pil_images[0].save(
                filename,
                save_all=True,
                append_images=pil_images[1:],
                duration=self.timer.interval(),
                loop=0
            )
            print(f"Animation successfully saved as {filename}")
        except Exception as e:
            print(f"Error saving GIF: {e}")


class AnimatedGraph(FramesWindow):
    def __init__(self, x_data, y_data, title="Transit Light Curve", parent=None, size: tuple[int] = (600, 600), x_label = 'Phase', y_label = 'Magnitude Change', invert_x: bool = False, invert_y: bool = True, legend: str = 'model', modal: bool = True):
        self.x_label = x_label
        self.y_label = y_label
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.legend = legend
        frames = self._generate_highlight_frames(x_data, y_data)

        super().__init__(frames, parent, title, size, modal)

    def _generate_highlight_frames(self, x_data, y_data) -> list[QImage]:
        """
        Generates a list of QImage frames from a Matplotlib plot,
        with each frame highlighting a different data point.
        """
        frames = []
        num_points = len(x_data)

        for i in range(num_points):
            # Create a Matplotlib figure for each frame
            fig, ax = plt.subplots(figsize=(6, 6))

            # Plot all points in blue
            ax.plot(x_data, y_data, label=self.legend)

            # Highlight the i-th point in red
            ax.plot(x_data[i], y_data[i], 'ro', markersize=10, label=f'current phase {(i + 1) / num_points}')

            if self.invert_x:
                ax.invert_xaxis()

            if self.invert_y:
                ax.invert_yaxis()

            ax.set_xlabel(self.x_label)
            ax.set_ylabel(self.y_label)
            ax.legend()

            # Convert the Matplotlib figure to a QImage
            # Use a BytesIO object to capture the figure as a PNG
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)  # Close the figure to free up memory
            buf.seek(0)

            # Create a QImage from the PNG data
            image = QImage.fromData(buf.getvalue())
            frames.append(image)

        return frames

class AnimationWindow(FramesWindow):
    """transit animation window (opened using the "Show transit animation" window)"""
    def __init__(self, star, exoplanet, parent=None, title: str = "Ringed Exoplanet Transit Simulation", size: tuple[int] = (600, 600), modal: bool = True):
        super().__init__(star.transit_frames(exoplanet), parent, title, size, modal)

app = QApplication(sys.argv)
