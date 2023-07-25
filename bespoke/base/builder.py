from abc import ABC, abstractmethod

from bespoke import backend as B
from bespoke.base.topology import Node

class HouseBuilder(ABC):
    """ Model House Builder

    """

    def __init__(self, model_house):
        self._model_house = model_house
    
    @abstractmethod
    def build(self, num):
        """Generate an alternative network for `net`

        """

class RandomHouseBuilder(HouseBuilder):
    """ Random-based Model House Builder

    """

    def __init__(self, model_house):
        super(RandomHouseBuilder, self).__init__(model_house)

    def build(self, num, step_ratio=0.1):
        nets = self._model_house.parser.get_random_subnets(num=num, step_ratio=step_ratio)
        for net, input_, output_ in nets:
            n = Node(self._model_house._parser.get_id("node"), "origin", net, pos=(tuple(input_), tuple(output_)), is_original=True)
            self._model_house.add(n)
            n.sleep()
