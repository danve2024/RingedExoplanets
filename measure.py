from typing import Union

# A class used for creating parameter ranges for the model

class Measure:
    """
    Measure.min - minimum parameter value
    Measure.max - maximum parameter value
    Measure.unit - unit of measurement for the parameter
    Measure.range - range of values for the parameter
    Measure.label - label for printing the parameter values
    Measure.prefix - prefix for printing the parameter values
    Measure.Unit - a class used for converting between different units of measurement for the parameter
    """
    def __init__(self, minimum: float, maximum: float, unit: Union[int, float] = 1, label: str = None, print_values=False):
        """
        Creates a parameter range for the model.

        :param float minimum: Minimum value of the parameter.
        :param float maximum: Maximum value of the parameter.
        :param unit: Unit of measurement for the parameter.
        :param str label: Label for the parameter.
        :param bool print_values: Whether to print the initial values of the parameter.
        """
        self.min = self.Unit(minimum).set(unit)
        self.max = self.Unit(maximum).set(unit)
        self.mean = (self.min + self.max) / 2
        self.unit = self.Unit(unit)
        self.range = self.max - self.min
        self.label = label
        if self.label:
            self.prefix = self.label + ': '
        else:
            self.prefix = ''
        if print_values:
            print(self)

    def slider(self, value: int):
        """
        Converts a slider value to a parameter value for the visualization.

        :param int value: Slider value (0-100).
        :return: Parameter value.
        """
        return self.min(self.unit) + self.range(self.unit) * (value / 100)

    def update(self, minimum: float, maximum: float):
        """
        Updates the parameter range.

        :param float minimum: New minimum value of the parameter.
        :param float maximum: New maximum value of the parameter
        """
        self.__init__(minimum, maximum, self.unit, self.label)

    def __contains__(self, item):
        return self.min <= item <= self.max

    def __str__(self):
        """
        Prints the parameter range.

        :return: String representation of the parameter range.
        """
        return f'{self.prefix}measure({self.min}, {self.max}, {self.unit})'

    class Unit(float):
        """
        A class used for converting between different units of measurement for the parameter.
        Special methods:
        Measure.Unit.__call__(unit) - converts a value to a specific unit of measurement
        Measure.Unit.set(unit) - writes a value from a specific unit of measurement
        """
        def __add__(self, other):
            return Measure.Unit(super().__add__(float(other)))

        def __sub__(self, other):
            return Measure.Unit(super().__sub__(float(other)))

        def __mul__(self, other):
            if isinstance(other, Measure.Unit):
                ans = super().__mul__(float(other))
            else:
                ans = super().__mul__(other)
            try:
                return Measure.Unit(ans)
            except TypeError:
                return ans

        def __truediv__(self, other):
            if isinstance(other, Measure.Unit):
                ans = super().__truediv__(float(other))
            else:
                ans = super().__truediv__(other)
            try:
                return Measure.Unit(ans)
            except TypeError:
                return ans

        def __pow__(self, other):
            return Measure.Unit(super().__pow__(float(other)))

        def __radd__(self, other):
            return Measure.Unit(super().__add__(float(other)))

        def __rsub__(self, other):
            return Measure.Unit(Measure.Unit(other).__sub__(float(self)))

        def __rmul__(self, other):
            return Measure.Unit(super().__mul__(float(other)))

        def __rtruediv__(self, other):
            return Measure.Unit(Measure.Unit(other).__truediv__(float(self)))

        def __rpow__(self, other):
            return Measure.Unit(Measure.Unit(other).__pow__(float(self)))

        def __floordiv__(self, other):
            return round(self/other) + 1

        def __call__(self, n):
            return self/n

        def set(self, n):
            return self*n