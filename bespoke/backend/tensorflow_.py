
from bespoke.backend import common

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation import parse, inject, cut, unfold

class TFParser(common.Parser):

    def __init__(self, model, custom_objects=None):
        self._parser = PruningNNParser(model, custom_objects=custom_objects)
        self._parser.parse()

    def build(self)

        min_layers = 5

        # construct t-rank
        v = self._parser.traverse()
        trank = {
            name:idx
            for idx, (name, _) in enumerate(v)
        }
        rtrank = {
            idx:name
            for name, idx in trank.items()
        }
        joints = set(self._parser.get_joints())

        groups = self._parser.get_sharing_groups()
        groups_ = []
        r_groups = {}
        for group in groups:
            group_ = []
            for layer_name in group:
                group_.append(self._model.get_layer(layer_name))
                r_groups[layer_name] = group_
            groups_.append(group_)

        def compute_constraints(layers):
            constraints = []
            for layer in layers:
                if layer.name in r_groups:
                    is_already = False
                    for c in constraints:
                        if c == r_groups[layer.name]:
                            is_already = True
                            break
                    if not is_already:
                        constraints.append(r_groups[layer.name])
            return constraints

        def is_compressible(layers):
            compressible = False
            for layer in layers:
                if layer.__class__.__name__ in ["Conv2D", "Dense", "DepthwiseConv2D"]:
                    compressible = True
                    break
            return compressible

        subnets = []
        layers_ = []
        for idx in range(len(trank)):
            name = rtrank[idx]
            if len(layers_) > min_layers and name in joints:
                if is_compressible(layers_):
                    # make subnet from layers_                 
                    subnet = self._parser.get_subnet(layers_, self._model) 
                    constraints = compute_constraints(layers_)
                    module = (subnet[0], subnet[1], subnet[2], constraints, self.namespace)
                    subnets.append(module)
                layers_ = []
            else:
                layers_.append(self._model.get_layer(name))
        if len(layers_) > 0 and is_compressible(layers_):
            constraints = compute_constraints(layers_)
            subnet = self._parser.get_subnet(layers_, self._model)
            module = ModuleHolder(subnet[0], subnet[1], subnet[2], constraints, self.namespace)
            self._modules.append(module)

    def make_train_graph()

