from abc import ABC, abstractmethod
import random

from bespoke import backend as B

class Generator(ABC):

    def __init__(self, namespace):
        self._namespace = namespace
    
    @abstractmethod
    def generate(self, net):
        """Generate an alternative network for `net`

        """

"""
class MergedGenerator(Generator):
    
    def __init__(self, namespace, generators):
        super(MergedGenerator, self).__init__(namespace)
        self.generators = generators

    def generate(self, net, custom_objects):
"""     
         
MAX_TRY = 20

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

class PretrainedModelGenerator(Generator):

    def __init__(self, namespace, mode=0, avoid=None):
        super(PretrainedModelGenerator, self).__init__(namespace)
        self.models = B.generate_pretrained_models()
        for m in self.models:
            m.build()

    def generate(self, net, num=5, memory_limit=None):
        # TODO: single input shape
        input_shape = net.input_shapes[0]
        output_shape = net.output_shapes[0]

        ret = []
        num_try = 0
        while len(ret) < num:
            ridx = random.randint(0, len(self.models)-1)
            tmodel_parser = self.models[ridx]
            nets = tmodel_parser.get_random_subnets(num=1, target_shapes=(input_shape, output_shape), memory_limit=memory_limit)
            num_try += 1
            if nets is not None:
                ret.extend([n[0] for n in nets])
            if num_try > MAX_TRY:
                break
        return ret
