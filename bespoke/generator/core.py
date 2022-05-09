from abc import ABC, abstractmethod
import random

from bespoke import backend as B
from bespoke import config

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

class PretrainedModelGenerator(Generator):

    def __init__(self, namespace, model_list=None):
        super(PretrainedModelGenerator, self).__init__(namespace)
        self.models = B.generate_pretrained_models(model_list)
        for m in self.models:
            m.build()

    def generate(self, net, num=5, memory_limit=None, params_limit=None):
        # TODO: single input shape
        input_shape = net.input_shapes[0]
        output_shape = net.output_shapes[0]

        ret = []
        num_try = 0
        while len(ret) < num:
            ridx = random.randint(0, len(self.models)-1)
            tmodel_parser = self.models[ridx]
            nets = tmodel_parser.get_random_subnets(
                num=1,
                target_shapes=(input_shape, output_shape),
                memory_limit=memory_limit,
                params_limit=params_limit)
            num_try += 1
            if nets is not None:
                ret.extend([(n[0], self.models[ridx].__class__.__name__) for n in nets])
            if num_try > config.MAX_TRY:
                break
        return ret
