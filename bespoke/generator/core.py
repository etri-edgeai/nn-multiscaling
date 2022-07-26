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

    def generate(self, net, scales=None, init=False, custom_objects=None):
        if scales is None:
            scales = [0.25, 0.5, 0.75]

        alternatives = []
        for idx, scale in enumerate(scales):
            alter = B.prune(net, scale, self._namespace, init=init, custom_objects=custom_objects)
            alternatives.append(alter)
        return alternatives

class PretrainedModelGenerator(Generator):

    def __init__(self, namespace, model_list=None):
        super(PretrainedModelGenerator, self).__init__(namespace)
        self.models = B.generate_pretrained_models(model_list)
        for m in self.models:
            m.build()

    def generate(self, net, last=None, num=1, memory_limit=None, params_limit=None, sample_data=None, step_ratio=0.1, use_adapter=False):
        # TODO: single input shape
        input_shape = net.input_shapes[0]
        output_shape = net.output_shapes[0]

        ret = []
        num_try = 0
        history = set()
        while len(ret) < num:
            ridx = random.randint(0, len(self.models)-1)
            tmodel_parser = self.models[ridx]
            nets = tmodel_parser.get_random_subnets(
                num=1,
                target_shapes=(input_shape, output_shape),
                target_type=B.get_type(net.model, last),
                memory_limit=memory_limit,
                params_limit=params_limit,
                history=history,
                use_prefix=True,
                use_random_walk=False,
                sample_data=sample_data,
                step_ratio=step_ratio,
                use_adapter=use_adapter)
            num_try += 1
            if nets is not None:
                ret.extend([(n[0], self.models[ridx].get_model_name()) for n in nets])
            if num_try > config.MAX_TRY:
                break
        return ret
