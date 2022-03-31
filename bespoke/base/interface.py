from __future__ import absolute_import
from __future__ import print_function

from bespoke import backend as B

class ModelHouse(object):
    """This is a class of housing a model for model scaling.

    """

    def __init__(self, model, custom_objects=None):
        model_, parser = B.preprocess(model, custom_objects)
        self._model = model_
        self._parser = parser
        self._namespace = set()
        self._root = None

    def build(self):
        self._nodes = self._parser.build(self._model)

    def extract(self, recipe):
        pass

    def query(self):
        pass

    def profile(self):
        pass

    def dump(self):
        pass
    
    def load(self):
        pass

    def add(self):
        pass

    def remove(self):
        pass

