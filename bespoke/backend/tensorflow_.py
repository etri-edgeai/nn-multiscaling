from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation import parse, inject, cut, unfold
from nncompress.backend import add_prefix

from bespoke.backend import common
from bespoke.base.topology import Node, Edge, BranchingNode, AlternativeNode, PositionEdge

def preprocess(model, namespace, custom_objects):
    model = unfold(model, custom_objects)
    return model, TFParser(model, namespace, custom_objects)

class TFParser(common.Parser):

    def __init__(self, model, namespace, custom_objects=None):
        super(TFParser, self).__init__(model, namespace)
        self._parser = PruningNNParser(model, custom_objects=custom_objects, namespace=namespace)
        self._parser.parse()

    def build(self):

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

        root_node = BranchingNode(self._parser.get_id("root"), TFNet(self._model)) 
        layers_ = []
        for idx in range(len(trank)):
            name = rtrank[idx]
            if len(layers_) > min_layers and name in joints:
                if is_compressible(layers_):
                    # make subnet from layers_                 
                    subnet = self._parser.get_subnet(layers_, self._model) 
                    constraints = compute_constraints(layers_)
                    net = TFNet(subnet[0])
                    root_node.add(self._parser.get_id("bnode"), net, pos=(subnet[1], subnet[2]))
                layers_ = []
            else:
                layers_.append(self._model.get_layer(name))
        if len(layers_) > 0 and is_compressible(layers_):
            constraints = compute_constraints(layers_)
            subnet = self._parser.get_subnet(layers_, self._model)
            net = TFNet(subnet[0])
            root_node.add(self._parser.get_id("bnode"), net, pos=(subnet[1], subnet[2]))
        return root_node


class TFNet(common.Net):

    def __init__(self, model):
        super(TFNet, self).__init__(model)
    
    @property
    def input_shapes(self):
        if type(self._model.input) == list:
            return [s.shape for s in self._model.input]
        else:
            return self._model.input.shape

    def predict(self, data):
        return self._model(data)


def make_train_model(root, scale=0.1, range_=None):

    outputs = []
    outputs.extend(root.net.model.outputs)
    output_map = []

    _inputs = [input_ for input_ in root.net.model.inputs]
    _outputs = [output_ for output_ in root.net.model.outputs]
    stk = [(root, _inputs, _outputs)]

    output_idx = {}

    _cnt = 0
    while len(stk) != 0:
        curr, inputs_, outputs_ = stk.pop()

        # update t_inputs
        if type(curr) == BranchingNode:
            t_inputs = inputs_
            t_outputs = outputs_
            _cnt += 1
        elif type(curr) == AlternativeNode:

            if range_ is not None and (range_[0] > _cnt or _cnt > range_[1]):
                continue

            if type(curr.net.model.inputs) == list:
                out_ = curr.net.model(t_inputs)
            else:
                out_ = curr.net.model(t_inputs[0])

            if type(out_) != list:
                t_out = t_outputs[0]
            else:
                t_out = t_outputs

            if t_out.name not in output_idx:
                outputs.append(t_out)
                output_idx[t_out.name] = len(outputs)-1

            if out_.name not in output_idx:
                outputs.append(out_)
                output_idx[out_.name] = len(outputs)-1

            output_map.append((t_out, out_))

        else:
            raise NotImplementedError("Node type error")

        for e in curr.children:
            child = e.node

            # branching
            if type(e) == PositionEdge:
                _inputs, _outputs = e.pos
                _inputs = [root.net.model.get_layer(layer).output for layer in _inputs]
                _outputs = [root.net.model.get_layer(layer).output for layer in _outputs]
            else:
                _inputs, _outputs = (None, None)

            stk.append((child, _inputs, _outputs))
 
    house = tf.keras.Model(root.net.model.inputs, outputs) # for test
    #model = unfold(house, custom_objects={"SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer})
    #tf.keras.utils.plot_model(model, to_file="unfold_house.pdf", show_shapes=True)
    #house = model

    for (t, s) in output_map:
        house.add_loss(tf.reduce_mean(tf.keras.losses.mean_squared_error(t, s)*scale))

    return house, output_idx, output_map


def prune(net, scale, namespace, custom_objects=None):

    # TODO: needs improvement
    key = "pp"
    hit = None
    for i in range(1000):
        if key+str(i) in namespace:
            continue
        else:
            hit = key+str(i)
            namespace.add(hit)
            break

    pmodel = add_prefix(net.model, hit, custom_objects=custom_objects)
    parser = PruningNNParser(pmodel, custom_objects=custom_objects, gate_class=SimplePruningGate, namespace=namespace)
    parser.parse()

    gated_model = parser.inject(with_splits=True, allow_pruning_last=True)
    for layer in gated_model.layers:
        if type(layer) == SimplePruningGate:
            layer.collecting = False
            layer.data_collecting = False

    last_transformers = parser.get_last_transformers()

    # initialize gates
    for t, g in parser.get_t2g().items():
        t = gated_model.get_layer(t)
        w = t.get_weights()[0]
        if t in last_transformers:
            keep = np.oneslike((w.shape[-1],))
        else:
            w = np.abs(w)
            sum_ = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
            sorted_ = np.sort(sum_, axis=None)
            val = sorted_[int((len(sorted_)-1)*scale)]
            keep = (sum_ >= val).astype(np.float32)
        gate_layer = gated_model.get_layer(g)
        gate_layer.set_weights([keep])

    return TFNet(gated_model)
