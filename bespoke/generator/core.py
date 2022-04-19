from abc import ABC, abstractmethod

from bespoke import backend as B

class Generator(ABC):

    def __init__(self, namespace):
        self._namespace = namespace
    
    @abstractmethod
    def generate(self, net):
        """Generate an alternative network for `net`

        """

class PruningGenerator(Generator):

    def __init__(self, namespace):
        super(PruningGenerator, self).__init__(namespace)

    def generate(self, net, scales=None, custom_objects=None):
        if scales is None:
            scales = [0.25, 0.5, 0.75]

        alternatives = []
        for idx, scale in enumerate(scales):
            alter = B.prune(net, scale, self._namespace, custom_objects)
            alternatives.append(alter)
        return alternatives
