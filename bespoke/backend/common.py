from abc import ABC, abstractmethod

class Parser(ABC):

    @abstractmethod
    def build(self, model):
        """This function parses a model to make a model house.

        """
