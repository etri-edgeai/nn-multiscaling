""" Alternative model generator

"""


from abc import ABC, abstractmethod
import random

from bespoke import backend as B
from bespoke import config

class Generator(ABC):
    """ Alternative model generator - Base

    """

    def __init__(self, namespace):
        """ Init function

        """
        self._namespace = namespace
    
    @abstractmethod
    def generate(self, net):
        """Generate an alternative network for `net`

        """

class PruningGenerator(Generator):
    """ Making alternatives by pruning

    """

    def __init__(self, namespace):
        """ Init function

        """
        super(PruningGenerator, self).__init__(namespace)

    def generate(self, net, scales=None, init=False, sample_data=None, pruning_exit=False, custom_objects=None):
        """ Generate alternatives by pruning

        """
        if scales is None:
            scales = [0.125, 0.25, 0.5, 0.75]

        alternatives = []
        for idx, scale in enumerate(scales):
            if sample_data is None or random.random() > 0.5:
                alter = B.prune(net, scale, self._namespace, init=init, custom_objects=custom_objects)
            else:
                alter = B.prune_with_sampling(
                    net,
                    scale,
                    self._namespace,
                    sample_data,
                    init=init,
                    pruning_exit=pruning_exit,
                    custom_objects=custom_objects)
            if not alter:
                return False
            alternatives.append(alter)
        return alternatives

class PretrainedModelGenerator(Generator):
    """ Get prtrained models

    """

    def __init__(self, namespace, model_list=None):
        """ Init function

        """
        super(PretrainedModelGenerator, self).__init__(namespace)
        self.models = B.generate_pretrained_models(model_list)
        for m in self.models:
            m.build()

    def generate(self,
        net,
        last=None,
        num=1,
        memory_limit=None,
        params_limit=None,
        sample_data=None,
        step_ratio=0.1,
        use_adapter=False,
        use_last_types=False):
        """ Generate/find random subnetworks of pretrained models as alternatives

        """
        # TODO: single input shape
        input_shape = net.input_shapes[0]
        output_shape = net.output_shapes[0]

        if not use_last_types:
            last = B.get_type(net.model, last)

        ret = []
        num_try = 0
        history = set()
        while len(ret) < num:
            ridx = random.randint(0, len(self.models)-1)
            tmodel_parser = self.models[ridx]
            nets = tmodel_parser.get_random_subnets(
                num=1,
                target_shapes=(input_shape, output_shape),
                target_type=last,
                memory_limit=memory_limit,
                params_limit=params_limit,
                history=history,
                use_prefix=True,
                use_random_walk=False,
                sample_data=sample_data,
                step_ratio=step_ratio,
                use_adapter=use_adapter,
                use_last_types=use_last_types)
            num_try += 1
            if nets is not None:
                ret.extend([(n[0], self.models[ridx].get_model_name()) for n in nets])
            if num_try > config.MAX_TRY:
                break
        return ret
