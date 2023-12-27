""" Base Structure

"""

from abc import ABC, abstractmethod

class Parser(ABC):
    """ Base Parser for a model

    """

    def __init__(self, model, namespace):
        """ Init function

        """
        self._model = model
        self._namespace = namespace

    @abstractmethod
    def build(self):
        """This function parses a model to make a model house.

        """

    @abstractmethod
    def get_id(self, prefix):
        """ Get identifier for a prefix

        """


class Net(ABC):
    """ Wrapper for a model

    """

    def __init__(self, model):
        """ Init function

        """
        self._model = model

    @property
    def model(self):
        """ Getter for model

        """
        return self._model

    @model.setter
    def model(self, model):
        """ Setter for model

        """
        self._model = model

    @abstractmethod
    def predict(self, data):
        """This function provides an abstraction for a model's prediction.

        """

    @property
    @abstractmethod
    def input_shapes(self):
        """This function returns the shapes of inputs.

        """
    @abstractmethod
    def save(self, name, save_dir):
        """Save a network into a file

        """

    @classmethod
    @abstractmethod
    def load(self, filepath, custom_objects=None):
        """Load a network from a file

        """
