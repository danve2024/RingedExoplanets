import csv

class Observations:
    """
    A class for working with the observation data in CSV format.
    """
    def __init__(self, filename: str):
        """
        Converting the CSV data to a list of points for the graph for comparing with the model
        :param filename: CSV file
        """
        self.source_data = []

        with open(filename, 'r', encoding='utf-8-sig') as data:
            for line in csv.reader(data):
                self.source_data.append((float(line[0]), float(line[1])))

        self.data = self.source_data.copy()

    def __str__(self):
        return str(self.data)

    def shift(self, period: float, time_shift: float, magnitude_shift: float):
        """
        See formulas 2.4.1 - 2.4.3.
        Normalizes and moves the source data along the axes. It is needed for working with the magnitude change (produced by the model) instead of the magnitude itself (observed) and normalizing the time span (using phase instead of time).

        :param period: P
        :param time_shift: Δt
        :param magnitude_shift: m_0
        :return:
        """
        ending_point = time_shift + period # ending time, t_e
        new_data = []
        for i in self.source_data:
            if ending_point > i[0] > time_shift:
                new_data.append((i[0] - time_shift, i[1] + magnitude_shift))
        self.data = new_data

    def convert_hours(self):
        """
        Converts the time values in the source data from hours to seconds.
        """
        self.normalize_x(3600)

    def convert_minutes(self):
        """
        Converts the time values in the source data from minutes to seconds.
        """
        self.normalize_x(60)

    def convert_seconds(self):
        """
        Converts the time values in the source data from seconds to seconds.
        """
        pass

    def convert_milliseconds(self):
        """
        Converts the time values in the source data from milliseconds to seconds.
        """
        self.normalize_x(0.001)

    def convert_microseconds(self):
        """
        Converts the time values in the source data from microseconds to seconds.
        """
        self.normalize_x(10^(-6))

    def convert_kiloseconds(self):
        """
        Converts the time values in the source data from kiloseconds to seconds.
        """
        self.normalize_x(1000)

    def convert_megaseconds(self):
        """
        Converts the time values in the source data from megaseconds to seconds.
        """
        self.normalize_x(1000000)

    def normalize_x(self, time_span: float):
        """
        See formula 2.4.6.
        Normalizes the time, converting it into the phase parameter in the source data array.
        :param time_span: t_n
        :return:
        """
        new_data = []
        for i in self.source_data:
            new_data.append((i[0] / time_span, i[1]))
        self.source_data = new_data

    def crop_x(self, starting_point: float = 0., period: float = 1.):
        """
        See formulas 2.4.4 and 2.4.5.
        Crops the source data to represent the selected time period.
        :param starting_point: t_s
        :param period: P
        :return:
        """
        ending_point = starting_point + period # ending time (t_e)
        new_data = []
        for i in self.data:
            if ending_point > i[0] > starting_point:
                new_data.append((i[0] - period, i[1]))
        self.source_data = new_data

    def move_y(self, delta: float):
        """
        See formula 2.4.3.
        Shifts the magnitude by a specific value in the source data.
        :param delta: Δm
        :return:
        """
        new_data = []
        for i in self.data:
            new_data.append((i[0], i[1] + delta))
        self.source_data = new_data

def write_data(data: list, time_scale = 1, magnitude_shift = 0, time_shift = 0, filename: str = 'model.csv'):
    """
    Writes the observations data into a CSV file.
    """
    s = ''
    for i in data:
        s += f'{(i[0] + time_shift) * time_scale},{i[1] + magnitude_shift}\n'
    with open(filename, 'w', encoding='utf-8-sig') as file:
        file.write(s)
