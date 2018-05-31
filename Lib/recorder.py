"""Experiments records module."""


class Predictions:
    """Experiment simple round."""

    def __init__(self, classes):
        """Contructor."""
        self.__classes__ = classes
        self.__values__ = []


class Experiment:
    """Recorder simple element."""

    def __init__(self, params, comment=""):
        """Constructor."""
        self.__values__ = {}
        for i, p in enumerate(params):
            self.__values__[p] = []
        self.__comment__ = comment

    def add_values(self, values):
        """Add a row to __values__."""
        for key in values:
            for j in range(0, len(values[key])):
                self.__values__[key].append(values[key][j])

    def fill_with_history(self, constants, history):
        """Fill the __values__ with the keras History. Put constants first."""
        for i in range(0, len(history["acc"])):
            self.__values__["epoch"].append(i)
            for key in constants:
                self.__values__[key].append(constants[key])
            for key in history:
                self.__values__[key].append(history[key][i])



class Recorder:
    """Recorder."""

    def __init__(self, folderpath):
        """Constructor."""
        self.__folderpath__ = folderpath
        self.__experiments__ = []

    def add_experiment(self, experiment):
        """Add a row to __experiments."""
        self.__experiments__.append(experiment)
