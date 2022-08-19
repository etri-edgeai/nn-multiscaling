from __future__ import absolute_import
from __future__ import print_function

import os
import random
import json
import copy
import tempfile
import pickle

from numba import jit
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from keras_flops import get_flops

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation import parse, inject, cut, unfold
from nncompress.backend import add_prefix
from tqdm import tqdm

from bespoke.backend import common
from bespoke.base.topology import Node
from bespoke import config

STOP_POINTS = [
    tf.keras.layers.Activation,
    tf.keras.layers.ReLU,
    tf.keras.layers.BatchNormalization,
    tf.keras.layers.Add,
    tf.keras.layers.Concatenate,
    tf.keras.layers.Dropout,
    tf.keras.layers.Multiply
]

def preprocess(model, namespace, custom_objects):
    model = unfold(model, custom_objects)
    return model, TFParser(model, namespace, custom_objects)

def equivalent(a, b):

    if a.__class__.__name__ == "Activation":
        a = a.activation
    if b.__class__.__name__ == "Activation":
        b = b.activation
    if a.__class__.__name__ not in ["type", "function"]:
        a = a.__class__
    if b.__class__.__name__ not in ["type", "function"]:
        b = b.__class__

    a = a.__name__.lower()
    b = b.__name__.lower()

    if a == b:
        return True
    elif a in b:
        return True
    elif b in a:
        return True
    elif a in ["relu", "swish"] and b in ["relu", "swish"]:
        return True
    else:
        return False

class TFParser(common.Parser):

    def __init__(self, model, namespace, custom_objects=None):
        super(TFParser, self).__init__(model, namespace)
        self._parser = PruningNNParser(model, custom_objects=custom_objects, namespace=namespace, gate_class=SimplePruningGate)
        self._parser.parse()

        self._joints = None
        self._groups = None
        self._r_groups = None
        self._trank = None
        self._rtrank = None
        self.build()

    def get_id(self, prefix):
        return self._parser.get_id(prefix)

    def get_model_name(self):
        return self._parser.model.name

    def _compute_constraints(self, layers):
        constraints = []
        for layer in layers:
            if layer.name in self._r_groups:
                is_already = False
                for c in constraints:
                    if c == self._r_groups[layer.name]:
                        is_already = True
                        break
                if not is_already:
                    constraints.append(self._r_groups[layer.name])
        return constraints

    def _is_compressible(self, layers):
        compressible = False
        for layer in layers:
            if layer.__class__.__name__ in ["Conv2D", "Dense", "SeparableConv2D"]:
                compressible = True
                break
        return compressible

    def is_compatible(self, a, b):
        pos_a = a.pos
        pos_b = b.pos

        irank_a = self._trank[pos_a[0][0]]
        orank_a = self._trank[pos_a[1][0]]
        irank_b = self._trank[pos_b[0][0]]
        orank_b = self._trank[pos_b[1][0]]

        if irank_a >= orank_b:
            return True
        if orank_a <= irank_b:
            return True
        return False

    def extract(self, origin_nodes, maximal, return_gated_model=False):
        return extract(self._parser, origin_nodes, maximal, self._trank, return_gated_model=return_gated_model)

    def get_random_subnets(
        self, num=1, target_shapes=None, target_type=None, memory_limit=None, params_limit=None, step_ratio=0.1, batch_size=32, history=None, use_prefix=False, use_random_walk=False, sample_data=None, use_adapter=False):

        def f(node_dict):
            return self._model.get_layer(node_dict["layer_dict"]["name"]).__class__ in STOP_POINTS

        stops = [
            layer.name for layer in self._parser.model.layers if layer.__class__ in STOP_POINTS
        ]

        stops = sorted(stops, key=lambda x: self._trank[x])
        nets = []
        if history is None:
            history = set()
        for i in range(num):
            layers_ = []
            num_try = 0
            subnet = None
            while True:
                if num_try > config.MAX_TRY:
                    break
                num_try += 1

                r = stops[random.randint(0, len(stops)-5)]
                left = self._trank[r]
                if target_shapes is not None:
                    input_shape, output_shape = target_shapes
                    if type(self._model.get_layer(self._rtrank[left]).input) == list:
                        continue
                    left_shape = self._model.get_layer(self._rtrank[left]).input.shape
                    if (not input_shape[-1] <= left_shape[-1]) or (not use_adapter):
                        continue

                if not use_random_walk:
                    joints = self._parser.get_joints(filter_=f, start=r)

                    if len(joints) <= 1:
                        continue

                    # select two positions from the joints randomly
                    #left_idx = random.randint(0, len(joints)-2)
                    step = int(len(joints) * step_ratio)
                    if step == 0:
                        min_ = len(joints)-1
                    else:
                        min_ = min(len(joints)-1, step)

                    right_idx = random.randint(1, min_)
                    right = self._trank[joints[right_idx]]

                else:
                    trail = self._parser.get_randomwalk(
                        r, p=1.0, types=STOP_POINTS)
                    atrail = [
                        t
                        for t in trail if self._parser.model.get_layer(t).__class__ in STOP_POINTS
                    ]

                    if len(atrail) <= 1:
                        continue

                    step = int(len(atrail) * step_ratio)
                    if step == 0:
                        min_ = len(atrail)-1
                    else:
                        min_ = min(len(atrail)-1, step) # left_idx = 0

                    right_idx = random.randint(1, min_)
                    right = self._trank[atrail[right_idx]]

                if (self._parser.model.name, left, right) in history:
                    continue
                history.add((self._parser.model.name, left, right))

                if target_type is not None and not equivalent(self._model.get_layer(self._rtrank[right]), target_type):
                    continue

                if target_shapes is not None:
                    input_shape, output_shape = target_shapes
                    if type(self._model.get_layer(self._rtrank[left]).input) == list:
                        continue
                    if type(self._model.get_layer(self._rtrank[right]).output) == list:
                        continue

                    left_shape = self._model.get_layer(self._rtrank[left]).input.shape
                    right_shape = self._model.get_layer(self._rtrank[right]).output.shape

                    # scale test
                    spatial_change = output_shape[1] / input_shape[1]
                    channel_change = output_shape[-1] / input_shape[-1]
                    
                    target_schange = right_shape[1] / left_shape[1]
                    target_cchange = right_shape[-1] / left_shape[-1]

                    if not(spatial_change == target_schange and\
                        ((input_shape[-1] <= left_shape[-1] and output_shape[-1] <= right_shape[-1]) or use_adapter)):
                        continue

                layers_ = [
                    self._model.get_layer(self._rtrank[j])
                    for j in range(left, right+1)
                ]

                if not self._is_compressible(layers_):
                    continue

                # Assumption: # of inputs is 1.
                in_target_shapes_ = [target_shapes[0]] if target_shapes is not None else None
                out_target_shapes_ = [target_shapes[1]] if target_shapes is not None else None
                subnet = self._parser.get_subnet(layers_, self._model, in_target_shapes_, out_target_shapes_, use_adapter=use_adapter)
                if use_prefix:
                    prefix = subnet[0].name+"_"
                    subnet_ = [None for _ in range(3)]
                    subnet_[0] = add_prefix(subnet[0], subnet[0].name+"_", custom_objects=self._parser.custom_objects, not_change_model_name=True)
                    subnet_[1] = [
                        prefix+item for item in subnet[1]
                    ]
                    subnet_[2] = [
                        prefix+item for item in subnet[2]
                    ]
                    subnet = tuple(subnet_)

                if type(subnet[0].input) == list:
                    continue

                scales = [0.1, 0.125, 0.25, 0.375, 0.5, 0.625]
                pruning_cnt = len(scales)
                giveup = False
                while True:

                    if pruning_cnt < len(scales) and pruning_cnt > -1:
                        scale = scales[pruning_cnt]
                        print("Pruning! %f %d" % (scale, pruning_cnt))
                        subnet_ = prune(subnet[0], scale, self._namespace, custom_objects=self._parser.custom_objects, ret_model=True)
                        subnet_ = tuple([subnet_] + list(subnet)[1:])
                    elif pruning_cnt < 0:
                        giveup = True
                        break
                    
                    pass_ = True
                    if memory_limit is not None and memory_limit > 0.0:
                        shape = list(subnet_[0].input.shape)
                        shape[0] = batch_size
                        data = np.random.rand(*shape)
                        tf.config.experimental.reset_memory_stats('GPU:0')
                        peak1 = tf.config.experimental.get_memory_info('GPU:0')['peak']
                        try:
                            subnet_[0](data)
                        except Exception as e:
                            print("Extremely large! %d" % pruning_cnt)
                            print(target_shapes)
                            pruning_cnt -= 1
                            continue
                         
                        peak2 = tf.config.experimental.get_memory_info('GPU:0')['peak']
                        if memory_limit < peak2-peak1:
                            pass_ = False
                            print(memory_limit, peak2-peak1)

                    if params_limit is not None and params_limit > 0.0:
                        if subnet_[0].count_params() > params_limit:
                            pass_ = False

                    if not pass_:
                        pruning_cnt -= 1
                    else:
                        break

                if giveup:
                    continue
                else:
                    break

            if subnet is None:
                continue

            if target_shapes is not None and (subnet[0].output.shape[-1] < target_shapes[1][-1] or subnet[0].input.shape[-1] < target_shapes[0][-1]):
                print("lack of channels...")
                continue

            #constraints = self._compute_constraints(layers_)
            if target_shapes is not None and (subnet[0].output.shape[-1] > target_shapes[1][-1] or subnet[0].input.shape[-1] > target_shapes[0][-1]):

                print(subnet[0].output.shape[-1], subnet[0].input.shape[-1], target_shapes[0][-1], target_shapes[1][-1])
                subnet_parser = PruningNNParser(subnet[0], allow_input_pruning=True, gate_class=SimplePruningGate)
                subnet_parser.parse()
                groups, groups_top = subnet_parser.get_group_topology()

                fg = None
                lg = None
                input_mask = None
                output_mask = None
                if subnet[0].input.shape[-1] > target_shapes[0][-1]:
                    first = tuple(subnet_parser.get_first_transformers())
                    fg = None
                    fgtop = None
                    for g, gtop in zip(groups, groups_top):
                        if has_intersection(g, first):
                            fg = g
                            fgtop = gtop
                            break

                    score = [[i,0] for i in range(subnet[0].input.shape[-1])]
                    if fgtop is not None:
                        for key, val in fgtop.items():
                            if type(key) == str:
                                w = subnet[0].get_layer(key).get_weights()[0]
                                w = np.abs(w)
                                if len(w.shape) == 2:
                                    loc = (1,0)
                                elif len(w.shape) == 4:
                                    loc = (0,1,3,2)
                                else:
                                    raise NotImplementedError("w's shape: %d" % len(w.shape))
                                w = np.transpose(w, loc)
                                sum_ = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
                                dim = sum_.shape[-1]
                                for i in range(dim):
                                    for v in val:
                                        if v[0] <= i and i < v[1]:
                                            j = i - v[0]
                                            score[i][1] += sum_[j]

                    else: # Depthhwise Conv2D only case
                        for layer in subnet[0].layers:
                            if layer.__class__.__name__ == "DepthwiseConv2D":
                                w = layer.get_weights()[0]
                                w = np.abs(w)
                                if len(w.shape) == 2:
                                    loc = (1,0)
                                elif len(w.shape) == 4:
                                    loc = (0,1,3,2)
                                else:
                                    raise NotImplementedError("w's shape: %d" % len(w.shape))
                                w = np.transpose(w, loc)
                                sum_ = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
                                dim = sum_.shape[-1]
                                for i in range(dim):
                                    score[i][1] += sum_[i]
                                break

                    """
                    Considering shared convs with input layer.

                    for g, gtop in zip(groups, groups_top):
                        if has_intersection(g, tuple(subnet[0].input.name)):
                            sg = g
                            sgtop = gtop
                            break
                    """
                    score = sorted(score, key=lambda x:x[1], reverse=True)
                    input_mask = np.zeros((subnet[0].input.shape[-1]))
                    for cnt, (idx, s) in enumerate(score):
                        if cnt == target_shapes[0][-1]:
                            break
                        input_mask[idx] = 1.0

                if subnet[0].output.shape[-1] > target_shapes[1][-1]:
                    last = tuple(subnet_parser.get_last_transformers())
                    lg = None
                    lgtop = None
                    for g, gtop in zip(groups, groups_top):
                        if has_intersection(g, last):
                            lg = g
                            lgtop = gtop
                            break

                    score = [[i,0] for i in range(subnet[0].output.shape[-1])]
                    if lgtop is not None:
                        for key, val in lgtop.items():
                            if type(key) == str:
                                ws = subnet[0].get_layer(key).get_weights()
                                if len(ws) == 0: # input layer case
                                    continue
                                w = ws[0]
                                w = np.abs(w)
                                sum_ = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
                                dim = sum_.shape[-1]
                                for i in range(dim):
                                    for v in val:
                                        if v[0] <= i and i < v[1]:
                                            j = i - v[0]
                                            score[i][1] += sum_[j]

                    score = sorted(score, key=lambda x:x[1], reverse=True)
                    output_mask = np.zeros((subnet[0].output.shape[-1]))
                    for cnt, (idx, s) in enumerate(score):
                        if cnt == target_shapes[1][-1]:
                            break
                        output_mask[idx] = 1.0

                subnet_gmodel, gm = subnet_parser.inject(with_splits=True, with_mapping=True, allow_pruning_last=True)
                for layer in subnet_gmodel.layers:
                    if layer.__class__ == SimplePruningGate:
                        layer.gates.assign(np.ones((layer.ngates,)))
                if input_mask is not None:
                    for g, gtop in zip(groups, groups_top):
                        if has_intersection(g, [subnet[0].input.name]):
                            for l in g:
                                input_gate = subnet_gmodel.get_layer(gm[(l, 0)][0]["config"]["name"])
                                input_gate.gates.assign(input_mask)
                            break

                if output_mask is not None:
                    if lg is not None:
                        for g_ in lg:
                            output_gate = subnet_gmodel.get_layer(gm[(g_, 0)][0]["config"]["name"])
                            output_gate.gates.assign(output_mask)

                new_spatial_shape = {
                    subnet[0].input.name: input_shape
                }
                subnet_cmodel = subnet_parser.cut(subnet_gmodel, new_spatial_shape=new_spatial_shape)

                if subnet_cmodel.input.shape[-1] != input_shape[-1] or subnet_cmodel.output.shape[-1] != output_shape[-1]: # Due to complex dependency.. then skip.
                    del subnet_gmodel
                    del subnet_cmodel
                    continue

                assert subnet_cmodel.input.shape[-1] == target_shapes[0][-1]
                del subnet_gmodel

                if sample_data: # overflow test
                    y = subnet_cmodel(sample_data)
                    if tf.math.is_inf(y).numpy().any():
                        continue
            else:
                subnet_cmodel = subnet[0]

            net = TFNet(subnet_cmodel, custom_objects=self._parser.custom_objects)
            nets.append((net, subnet[1], subnet[2]))
            del subnet

        return nets

    def get_uniform_subsets(self, num=5):
        pass

    def build(self):
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
        def f(node_dict):
            return self._model.get_layer(node_dict["layer_dict"]["name"]).__class__ in STOP_POINTS
            #return len(node_dict["layer_dict"]["inbound_nodes"]) > 0 and len(node_dict["layer_dict"]["inbound_nodes"][0]) == 1
        joints = self._parser.get_joints(filter_=f)

        self._joints = joints
        self._trank = trank
        self._rtrank = rtrank

class TFNet(common.Net):

    def __init__(self, model, custom_objects=None):
        super(TFNet, self).__init__(model)
        self._custom_objects = custom_objects
        if custom_objects is None:
            self._custom_objects = {}
        if "SimplePruningGate" not in self._custom_objects:
            self._custom_objects["SimplePruningGate"] = SimplePruningGate
            self._custom_objects["StopGradientLayer"] = StopGradientLayer
        self.meta = {}

    @property
    def input_shapes(self):
        if self.is_sleeping():
            self.wakeup()
        if type(self._model.input) == list:
            return [s.shape for s in self._model.input]
        else:
            return [self._model.input.shape]

    @property
    def output_shapes(self):
        if self.is_sleeping():
            self.wakeup()
        if type(self._model.output) == list:
            return [s.shape for s in self._model.output]
        else:
            return [self._model.output.shape]

    def predict(self, data):
        if self.is_sleeping():
            self.wakeup()
        return self._model(data)

    def save(self, name, save_dir):
        if self.is_sleeping():
            self.wakeup()
        filepath = os.path.join(save_dir, name+".h5")
        ppath = os.path.join(save_dir, name+".pdf")
        tf.keras.utils.plot_model(self._model, ppath)
        tf.keras.models.save_model(self._model, filepath, overwrite=True)
        return filepath

    def profile(self, sample_inputs, sample_outputs, cmodel=None):
        if self.is_sleeping():
            self.wakeup()
        if cmodel is not None:
            return {
                "flops": get_flops(cmodel, batch_size=1),
                "mse": float(self.get_mse(sample_inputs, sample_outputs).numpy())
            }
        else:
            return {
                "flops": self.get_flops(),
                "mse": float(self.get_mse(sample_inputs, sample_outputs).numpy())
            }
        self.sleep()

    def get_flops(self):
        if self.is_sleeping():
            self.wakeup()
        flops = get_flops(self._model, batch_size=1)
        return flops

    def get_mse(self, sample_inputs, sample_outputs):
        if self.is_sleeping():
            self.wakeup()
        ret = 0
        for input_, output_ in zip(sample_inputs, sample_outputs):
            outputs = self._model(input_, training=False)
            ret += tf.reduce_mean(tf.keras.losses.mean_squared_error(output_, outputs))

        self.sleep()
        return ret

    def get_cmodel(self, origin_model):
        if self.is_sleeping():
            self.wakeup()
        flag = False
        for layer in self._model.layers:
            if layer.__class__ == SimplePruningGate:
                flag = True
                print(np.sum(layer.gates))
                print(layer.ngates)
                break
        if not flag:
            return self.model

        parser = PruningNNParser(origin_model, allow_input_pruning=True, custom_objects=self._custom_objects, gate_class=SimplePruningGate)
        parser.parse()
        cmodel = parser.cut(self._model)
        self.sleep()
        return cmodel

    def is_sleeping(self):
        return type(self._model) == tuple

    def sleep(self):
        json_ = self._model.to_json()
        weights = self._model.get_weights()
        model = self._model
        self._model = (json_, weights)
        del model

    def wakeup(self):
        if type(self._model) != tuple:
            return
        model = tf.keras.models.model_from_json(self._model[0], custom_objects=self._custom_objects)
        for layer in model.layers:
            if layer.__class__ == SimplePruningGate:
                layer.collecting = False
                layer.data_collecting = False
        model.set_weights(self._model[1])
        self._model = model

    @classmethod
    def load(self, filepath, custom_objects=None):
        model = load_model(filepath, custom_objects=custom_objects)
        for layer in model.layers:
            if layer.__class__ == SimplePruningGate:
                layer.collecting = False
                layer.data_collecting = False
        return TFNet(model, custom_objects)


def get_parser(model, namespace, custom_objects):
    return TFParser(model, namespace, custom_objects)


def extract(parser, origin_nodes, nodes, trank, return_gated_model=False):

    groups = parser.get_sharing_groups()
    r_groups = {}
    for group in groups:
        for layer_name in group:
            r_groups[layer_name] = group
    affecting = parser.get_affecting_layers()

    # get a pruned model with union consensus
    cparser = {}
    exit_gates = {}
    pruning_exit = False
    for n in nodes:
        if n.net.is_sleeping():
            n.net.wakeup()
        if n.origin is None:
            continue
        n.origin.net.wakeup()
        cparser[n.id_] = PruningNNParser(\
                n.origin.net.model, allow_input_pruning=True, custom_objects=n.origin.net._custom_objects, gate_class=SimplePruningGate)
        cparser[n.id_].parse()
        input_, output_ = n.pos
        
        if "gate_mapping" in n.net.meta:
            cgm = n.net.meta["gate_mapping"]
            pruning_exit = True
            last = cparser[n.id_].get_last_transformers()
            if type(last) == str:
                last = [last]
            for t in last:
                if t not in cgm:
                    continue
                gate_name = cgm[t][0]["config"]["name"]
                exit_gates[t] = n.net.model.get_layer(gate_name).gates.numpy()

    replacing_mappings = []
    in_maps = None
    ex_maps = []
    cmodel_map = {}
    pos_backup = [] # for debugging
    nodes = sorted(nodes, key=lambda x: trank[x.pos[0][0]])
    for idx, n in enumerate(nodes):
        print(idx, n.pos)
        if n.origin is None:
            continue
        pos = n.pos
        input_, output_ = pos

        # ones assigning
        backup = {}
        last = cparser[n.id_].get_last_transformers()
        if type(last) == str:
            last = [last]
        cgroups = cparser[n.id_].get_sharing_groups()

        for t in last:
            found = False
            for cg in cgroups:
                if t in cg:
                    found = True
                    break
            if not found:
                cgroups.append([t])

        for cg in cgroups:
            if n.net.model.input.name in cg:
                for l in cg:
                    if "gate_mapping" in n.net.meta:
                        gate_name = n.net.meta["gate_mapping"][l][0]["config"]["name"]
                        gates = n.net.model.get_layer(gate_name).gates
                        gates.assign(np.ones(gates.shape[-1],))

        for t in last:
            if "gate_mapping" in n.net.meta:
                for cg in cgroups:
                    if t in cg:
                        for l in cg:
                            gate_name = n.net.meta["gate_mapping"][l][0]["config"]["name"]
                            gates = n.net.model.get_layer(gate_name).gates
                            backup[gate_name] = gates.numpy()
                            gates.assign(np.ones(gates.shape[-1],))
                            print(gates.shape)
                        break
        
        cmodel = n.get_cmodel()
        cmodel_map[n.id_] = cmodel
        if n.net.is_sleeping():
            n.net.wakeup()

        # restore
        for t in last:
            if "gate_mapping" in n.net.meta:
                for cg in cgroups:
                    if t in cg:
                        for l in cg:
                            gate_name = n.net.meta["gate_mapping"][l][0]["config"]["name"]
                            gates = n.net.model.get_layer(gate_name).gates
                            gates.assign(backup[gate_name])
                        break

        onode = origin_nodes[tuple(n.pos)]
        onode.net.wakeup()
        omodel = onode.net.model

        # TODO: there is only one input layer
        target = []
        target.append(pos[0][0])
        for layer in omodel.layers:
            if layer.__class__ != tf.keras.layers.InputLayer:
                target.append(layer.name)

        prev_target = None
        prev_replacement = None
        for target_, replacement_ in replacing_mappings:
            if pos[0][0] in target_:
                prev_target = target_
                prev_replacement = replacement_
                break
        if prev_target is not None:
            _input_name = prev_replacement[-1]["name"]
            replacement = []
        else:
            _input = copy.deepcopy(parser.get_layer_dict(pos[0][0]))
            _input["inbound_nodes"] = []
            _input_name = _input["config"]["name"]
            replacement = [_input]

        inputs_ = []
        for layer in json.loads(cmodel.to_json())["config"]["layers"]:
            if layer["class_name"] != "InputLayer": # remove input layer
                replacement.append(layer)
            else:
                inputs_.append(layer["config"]["name"])

        for layer in replacement:
            for flow in layer["inbound_nodes"]:
                for inbound in flow:
                    if inbound[0] in inputs_:
                        inbound[0] = _input_name

        if prev_target is None or pos[0][0] not in prev_target: # Not Merge
            ex_map = [
                [(pos[0][0], _input_name)],
                [(pos[-1][0], replacement[-1]["name"], 0, 0)]
            ]
            ex_maps.append(ex_map)
            replacing_mappings.append((target, replacement))
            pos_backup.append(pos)
        else:
            for t in target:
                if t not in prev_target:
                    prev_target.append(t)
            prev_replacement += replacement
            ex_maps[-1][1].append((pos[-1][0], replacement[-1]["name"], 0, 0))

        n.net.sleep()
        onode.net.sleep()

    # Conduct replace_block
    model_dict = parser.replace_block(replacing_mappings, in_maps, ex_maps, parser.custom_objects)
    model_json = json.dumps(model_dict) 
    try:
        model = tf.keras.models.model_from_json(model_json, custom_objects=parser.custom_objects)
    except Exception as e:
        for ex_map, (target, replacement), pos in zip(ex_maps, replacing_mappings, pos_backup):
            print(ex_map)
            print(target)
            print(replacement)
            print(pos)
            print("---")
        import sys, traceback
        traceback.print_exc(file=sys.stdout)
        sys.exit()

    not_det = set()
    layer_names = [ layer.name for layer in model.layers ]
    for layer in parser.model.layers:
        if layer.name in layer_names:
            new_layer = model.get_layer(layer.name)
            try:
                new_layer.set_weights(layer.get_weights())
            except ValueError:
                not_det.add(layer.name)

    for _, cmodel_ in cmodel_map.items():
        for layer in cmodel_.layers:
            if layer.name in layer_names:
                new_layer = model.get_layer(layer.name)
                new_layer.set_weights(layer.get_weights())
                if layer.name in not_det:
                    not_det.remove(layer.name)
    assert len(not_det) == 0

    if not pruning_exit:
        if return_gated_model:
            return model, model, ex_maps
        else:
            return model
    else:
        parser_ = PruningNNParser(model, custom_objects=parser._custom_objects, gate_class=SimplePruningGate) 
        parser_.parse()
        gmodel, gm = parser_.inject(with_splits=True, with_mapping=True)

        for layer in gmodel.layers:
            if layer.__class__ == SimplePruningGate:
                layer.gates.assign(np.zeros((layer.ngates,)))

        sharing_groups = parser_.get_sharing_groups()
        sum_ratio = {}
        sum_cnt = {}
        for t in exit_gates:
            gates = exit_gates[t]
            target_gate = gmodel.get_layer(gm[(t, 0)][0]["config"]["name"])
            tgates = target_gate.gates.numpy()

            for gidx, g in enumerate(sharing_groups):
                if gidx not in sum_ratio:
                    sum_ratio[gidx] = 0.0
                    sum_cnt[gidx] = 0
                if t in g:
                    sum_ratio[gidx] = max(float(np.sum(gates)) / target_gate.ngates, sum_ratio[gidx])
                    sum_cnt[gidx] += 1
                    break
            #union = (((gates + tgates) == 0.0) == False).astype(np.float32)
            #target_gate.gates.assign(union)

        for t in exit_gates:
            gates = exit_gates[t]
            target_gate = gmodel.get_layer(gm[(t, 0)][0]["config"]["name"])
            #tgates = target_gate.gates.numpy()
            for gidx, g in enumerate(sharing_groups):
                if t in g:
                    sum_ = None 
                    for l in g:
                        layer = gmodel.get_layer(l)
                        if len(layer.get_weights()) > 0:
                            w = layer.get_weights()[0]
                            w = np.abs(w)
                            w = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
                            if sum_ is None:
                                sum_ = w
                            else:
                                sum_ += w
                    sorted_ = np.sort(sum_, axis=None)
                    #gscale = float(sum_ratio[gidx]) / sum_cnt[gidx]
                    gscale = float(sum_ratio[gidx])
                    remained = min(int((len(sorted_)-1)*gscale), len(sorted_)-1-5)
                    if remained >= len(sum_):
                        remained = len(sum_)-1
                    val = sorted_[remained]
                    mask = (sum_ >= val).astype(np.float32)

                    for l in g:
                        target_gate = gmodel.get_layer(gm[(l, 0)][0]["config"]["name"])
                        #target_gate.gates.assign(tgates)
                        target_gate.gates.assign(mask)

        for layer in gmodel.layers:
            if layer.__class__ == SimplePruningGate:
                if np.sum(layer.gates.numpy()) == 0:
                    layer.gates.assign(np.ones((layer.ngates,)))

        if return_gated_model:
            return gmodel, parser_.cut(gmodel), ex_maps
        else:
            ret = parser_.cut(gmodel)
            return ret
            
def make_train_model(model, nodes, scale=0.1, teacher_freeze=True):
    outputs = []
    outputs.extend(model.outputs)
    output_map = []

    _inputs = [input_ for input_ in model.inputs]
    _outputs = [output_ for output_ in model.outputs]

    if teacher_freeze:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in model.layers:
            layer.trainable = True

    output_idx = {}
    for n in nodes:
        t_inputs = [model.get_layer(layer).output for layer in n.pos[0]]
        t_outputs = [model.get_layer(layer).output for layer in n.pos[1]]

        if type(n.net.model.inputs) == list:
            out_ = n.net.model(t_inputs)
        else:
            out_ = n.net.model(t_inputs[0])

        for l in n.net.model.layers:
            l.trainable = True
            if l.__class__ == SimplePruningGate:
                l.collecting = False
                l.data_collecting = False
                #l.trainable = False

        if type(out_) != list:
            t_out = t_outputs[0]

            if t_out.name not in output_idx:
                outputs.append(t_out)
                output_idx[t_out.name] = len(outputs)-1

            if out_.name not in output_idx:
                outputs.append(out_)
                output_idx[out_.name] = len(outputs)-1

            output_map.append((t_out, out_))
        else:
            for t_out, a_out in zip(t_outputs, out_):

                if t_out.name not in output_idx:
                    outputs.append(t_out)
                    output_idx[t_out.name] = len(outputs)-1

                if a_out.name not in output_idx:
                    outputs.append(a_out)
                    output_idx[a_out.name] = len(outputs)-1

                output_map.append((t_out, a_out))

    house = tf.keras.Model(model.inputs, outputs) # for test

    #for (t, s) in output_map:
    #    house.add_loss(tf.reduce_mean(tf.keras.losses.mean_squared_error(t, s)*scale))

    return house, output_idx, output_map


def backend_net():
    return TFNet


def get_basemodel_path(dir_):
    return os.path.join(dir_, "base.h5")
    

def prune(net, scale, namespace, custom_objects=None, ret_model=False, init=False, pruning_exit=False):

    # TODO: needs improvement
    """
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
    """

    if type(net) == TFNet:
        parser = PruningNNParser(net.model, allow_input_pruning=pruning_exit, custom_objects=custom_objects, gate_class=SimplePruningGate, namespace=namespace)
    else:
        parser = PruningNNParser(net, allow_input_pruning=pruning_exit, custom_objects=custom_objects, gate_class=SimplePruningGate, namespace=namespace)
    parser.parse()

    gated_model, gm = parser.inject(with_splits=True, allow_pruning_last=pruning_exit, with_mapping=True)
    if init:
        gated_model = tf.keras.models.clone_model(gated_model)
    for layer in gated_model.layers:
        if type(layer) == SimplePruningGate:
            layer.collecting = False
            layer.data_collecting = False
            layer.gates.assign(np.ones((layer.ngates,)))

    last_transformers = parser.get_last_transformers()
    target_groups, gate_struct = parser.get_group_topology()
    for idx, (g, dict_) in enumerate(zip(target_groups, gate_struct)):
        if not pruning_exit:
            skip = False
            for key, val in dict_.items():
                if type(key) == str:
                    if key in last_transformers:
                        skip = True
                        break
            if skip:
                continue

        items = []
        max_ = 0
        for key, val in dict_.items():
            if type(key) == str:
                val = sorted(val, key=lambda x:x[0])
                items.append((key, val))
                for v in val:
                    if v[1] > max_:
                        max_ = v[1]
    
        mask = np.zeros((max_,))
        for key, val in items:
            layer = gated_model.get_layer(key)
            if len(layer.get_weights()) > 0:
                for v in val:
                    w = layer.get_weights()[0]
                    w = np.abs(w)
                    w = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
                    mask[v[0]:v[1]] += w

        sorted_ = np.sort(mask, axis=None)
        remained = max(int((len(sorted_)-1)*scale), 5)
        if remained >= len(mask):
            remained = len(mask)-1
        val_ = sorted_[remained]
        mask = np.zeros((max_,))
        for cidx in range(mask.shape[0]):
            if mask[cidx] < val_:
                avoid_prune = False
                for key, val in items:
                    if np.sum(mask[val[0][0]:val[0][1]]) <= 5.0:
                        avoid_prune = True
                        break
                if not avoid_prune:
                    mask[cidx] = 0.0
                else:
                    mask[cidx] = 1.0
            else:
                mask[cidx] = 1.0

        # Distribute
        for key, val in items:
            if len(val) > 1:
                for v in val[1:]:
                    mask[v[0]:v[1]] = mask[val[0][0]:val[0][1]]

        for key, val in dict_.items():
            if type(key) == str:
                gates = gated_model.get_layer(gm[(key,0)][0]["config"]["name"]).gates
                if gates.shape[0] != val[0][1] - val[0][0]:
                    return False
                gates.assign(mask[val[0][0]:val[0][1]])

    # test
    try:
        cutmodel = parser.cut(gated_model)
    except Exception as e:
        print(e)
        return False

    print(gated_model.count_params(), cutmodel.count_params())
    if gated_model.count_params() == cutmodel.count_params():
        print("fail!")
        return False

    if ret_model:
        return cutmodel
    else:
        print(gated_model.output.shape, cutmodel.output.shape)

    if not pruning_exit:
        net = TFNet(cutmodel, custom_objects=parser.custom_objects)
    else:
        net = TFNet(gated_model, custom_objects=parser.custom_objects)
        gate_mapping = {}
        for key, value in gm.items():
            # json serialization
            gate_mapping[key[0]] = value
        net.meta["gate_mapping"] = gate_mapping
    return net

def prune_with_sampling(net, scale, namespace, sample_data, custom_objects=None, ret_model=False, init=False, pruning_exit=False):

    # get inchannel dependency
    def get_vindices(lidx, group_struct):
        vindices = set([lidx])
        last = len(vindices)
        initial_run = True
        visited = set()
        while initial_run or (not last == len(vindices)):
            initial_run = False
            last = len(vindices)

            for key, val in group_struct.items():
                if type(key) == str and key not in visited:
                    val = sorted(val, key=lambda x:x[0])
                    for vidx in list(vindices):
                        found = False
                        for v in val:
                            if v[0] <= vidx and vidx < v[1]:
                                found = True
                                relative = vidx - v[0]
                                break
                        if found:
                            visited.add(key)
                            for v in val:
                                vindices.add(v[0]+relative)
        return vindices

    if type(net) == TFNet:
        parser = PruningNNParser(net.model, allow_input_pruning=pruning_exit, custom_objects=custom_objects, gate_class=SimplePruningGate, namespace=namespace)
    else:
        parser = PruningNNParser(net, allow_input_pruning=pruning_exit, custom_objects=custom_objects, gate_class=SimplePruningGate, namespace=namespace)
    parser.parse()

    gated_model, gm = parser.inject(with_splits=True, allow_pruning_last=pruning_exit, with_mapping=True)
    if init:
        gated_model = tf.keras.models.clone_model(gated_model)
    for layer in gated_model.layers:
        if type(layer) == SimplePruningGate:
            layer.collecting = False
            layer.data_collecting = False
            layer.gates.assign(np.ones((layer.ngates,)))

    Y = []
    for X in sample_data:
        Y.append(gated_model(X))

    # scoring
    last_transformers = parser.get_last_transformers()
    target_groups, gate_struct = parser.get_group_topology()
    n_channels = 0
    inverted = {}
    for idx, (g, dict_) in enumerate(zip(target_groups, gate_struct)):
        if not pruning_exit:
            skip = False
            for key, val in dict_.items():
                if type(key) == str:
                    if key in last_transformers:
                        skip = True
                        break
            if skip:
                continue

        max_ = 0
        for key, val in dict_.items():
            if type(key) == str:
                for v in val:
                    if v[1] > max_:
                        max_ = v[1]
        
        vindex = []
        for cidx in range(max_):
            check = False
            for v in vindex:
                if cidx in v:
                    check = True
            if check:
                continue
            v = get_vindices(cidx, dict_)
            vindex.append(v)

        for vidx in range(len(vindex)):
            inverted[n_channels] = (vindex[vidx], g, dict_, idx)
            n_channels += 1
        
    score = [0.0 for _ in range(n_channels)]
    for idx in tqdm(range(n_channels), ncols=80):
        v, g, dict_, _ = inverted[idx]
        max_ = 0
        for key, val in dict_.items():
            if type(key) == str:
                val = sorted(val, key=lambda x:x[0])
                for v_ in val:
                    if v_[1] > max_:
                        max_ = v_[1]
        mask = np.ones((max_,))
        for cidx in v:
            mask[cidx] = 0.0

        for key, val in dict_.items():
            if type(key) == str:
                gates = gated_model.get_layer(gm[(key,0)][0]["config"]["name"]).gates
                gates.assign(mask[val[0][0]:val[0][1]])

        sum_ = 0.0
        for X, y in zip(sample_data, Y):
            predicted = gated_model(X)
            sum_ += tf.math.reduce_mean(tf.keras.losses.mse(predicted, y))

        score[idx] = float(sum_)

        # restore
        mask = np.ones((max_,))
        for key, val in dict_.items():
            if type(key) == str:
                gates = gated_model.get_layer(gm[(key,0)][0]["config"]["name"]).gates
                gates.assign(mask[val[0][0]:val[0][1]])

    masks = {}
    for cnt in range(int(n_channels*scale)):
        
        minidx = np.argmax(score)

        v, g, dict_, idx = inverted[minidx]
        max_ = 0
        for key, val in dict_.items():
            if type(key) == str:
                val = sorted(val, key=lambda x:x[0])
                for v_ in val:
                    if v_[1] > max_:
                        max_ = v_[1]
        
        if idx not in masks:
            mask = np.ones((max_,))
            masks[idx] = mask
        else:
            mask = masks[idx]

        for cidx in v:
            mask[cidx] = 0.0

        if np.sum(mask) < 5.0:
            for cidx in v:
                mask[cidx] = 1.0
 
        score[minidx] = -999

    for idx, (g, dict_) in enumerate(zip(target_groups, gate_struct)):
        if idx not in masks:
            continue

        mask = masks[idx]
        # mask to gate
        for key, val in dict_.items():
            if type(key) == str:
                gates = gated_model.get_layer(gm[(key,0)][0]["config"]["name"]).gates
                if gates.shape[0] != val[0][1] - val[0][0]:
                    return False
                for v_ in val:
                    gates.assign(mask[v_[0]:v_[1]])
                    break
    # test
    try:
        cutmodel = parser.cut(gated_model)
    except Exception as e:
        print(e)
        return False


    print(gated_model.count_params(), cutmodel.count_params())
    if gated_model.count_params() == cutmodel.count_params():
        print("fail!")
        return False

    if ret_model:
        return cutmodel
    else:
        print(gated_model.output.shape, cutmodel.output.shape)

    if not pruning_exit:
        net = TFNet(cutmodel, custom_objects=parser.custom_objects)
    else:
        net = TFNet(gated_model, custom_objects=parser.custom_objects)
        gate_mapping = {}
        for key, value in gm.items():
            # json serialization
            gate_mapping[key[0]] = value
        net.meta["gate_mapping"] = gate_mapping
    return net


def save_model(name, model, save_dir):
    path = os.path.join(save_dir, name+".h5")
    tf.keras.models.save_model(model, path, overwrite=True)

def load_model_from_node(load_dir, id_, custom_objects=None):
    path = os.path.join(load_dir, id_+".h5")
    load_model(path, custom_objects)

def load_model(filepath, custom_objects=None):
    try:
        return tf.keras.models.load_model(filepath, custom_objects=custom_objects)
    except ValueError:
        if custom_objects is None:
            custom_objects = {}
        custom_objects["SimplePruningGate"] = SimplePruningGate
        custom_objects["StopGradientLayer"] = StopGradientLayer
        model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        for layer in model.layers:
            if layer.__class__ == SimplePruningGate:
                layer.collecting = False
                layer.data_collecting = False
                #layer.trainable = False
        return model

def generate_pretrained_models(list_=None):
    baselist = [
        tf.keras.applications.ResNet50V2,
        tf.keras.applications.InceptionResNetV2,
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.MobileNet,
        tf.keras.applications.DenseNet121,
        tf.keras.applications.NASNetMobile,
        tf.keras.applications.EfficientNetB1,
        tf.keras.applications.EfficientNetV2B1,
        tf.keras.applications.EfficientNetV2S,
        tf.keras.applications.resnet_rs.ResNetRS152,
        tf.keras.applications.resnet_rs.ResNetRS101,
        tf.keras.applications.regnet.RegNetX002,
        tf.keras.applications.regnet.RegNetX004
    ]
    dict_ = {
        class_.__name__:class_ for class_ in baselist
    }
    models = []
    if list_ is None:
        list_ = list(dict_.keys())
    for name in list_:
        if name in dict_:
            w = dict_[name]
            models.append(TFParser(w(include_top=True, weights="imagenet", classes=1000), namespace=set()))
    return models

def get_type(model, layer_name=None):
    if layer_name is None:
        return None
    if model.get_layer(layer_name).__class__ == tf.keras.layers.Activation:
        type_ = model.get_layer(layer_name).activation
    else:
        type_ = model.get_layer(layer_name).__class__
    return type_ 


def numpyfy(data):

    if type(data) == list:
        ret = []
        for d in data:
            ret.append(numpyfy(d))
        return ret
    else:
        return data.numpy()

def build_samples(model, data_gen, pos):
    outputs = [[],[]]
    for idx, p in enumerate(pos):
        for pp in p:
            outputs[idx].append(model.get_layer(pp).output)

    extractor = tf.keras.Model(model.inputs, outputs)
    results = []
    for data in data_gen:
        y = extractor(data, training=False)
        y = numpyfy(y)
        results += [y]
    return results

def cut(model, reference_model, custom_objects):
    parser = PruningNNParser(reference_model, custom_objects=custom_objects, gate_class=SimplePruningGate)
    parser.parse()
    return parser.cut(model)

def make_distiller(model, teacher, distil_loc, scale=0.1, model_builder=None):

    toutputs = []
    for loc in distil_loc:
        tlayer, _ = loc
        toutputs.append(teacher.get_layer(tlayer).output)
    toutputs.append(teacher.output)
    new_teacher = tf.keras.Model(teacher.input, toutputs)
    toutputs_ = new_teacher(model.get_layer("input_lambda").output)

    if model_builder is None:
        new_model = tf.keras.Model(model.input, [model.output]+toutputs_)
    else:
        new_model = model_builder(model.input, [model.output]+toutputs_)

    for idx, loc in enumerate(distil_loc):
        tlayer, layer = loc
        t = tf.cast(toutputs_[idx], tf.float32)
        s = tf.cast(model.get_layer(layer).output, tf.float32)
        new_model.add_loss(tf.reduce_mean(tf.keras.losses.mean_squared_error(t, s)*scale))
    new_model.add_loss(tf.reduce_mean(tf.keras.losses.kl_divergence(model.output, toutputs_[-1])*scale))

    for layer in teacher.layers:
        layer.trainable = False

    for layer in model.layers:
        layer.trainable = True
        if layer.__class__ == SimplePruningGate:
            #layer.trainable = False
            layer.collecting = False
            layer.data_collecting = False

    return new_model

def save_transfering_model(dirpath, house, output_idx, output_map):
    model_path = os.path.join(dirpath, "model.h5")
    output_idx_path = os.path.join(dirpath, "output_idx.pickle")
    output_map_path = os.path.join(dirpath, "output_map.pickle")

    tf.keras.models.save_model(house, model_path, overwrite=True)
    with open(output_idx_path, "wb") as f:
        pickle.dump(output_idx, f)

    with open(output_map_path, "wb") as f:
        output_map_ = [
            (a.name, b.name)
            for a, b in output_map
        ]
        pickle.dump(output_map_,f)
        
    return

def make_transfer_model(model, output_idx, output_map, scale, model_builder=None):

    if model_builder is not None:
        model = model_builder(model.input, model.output)

    for (t, s) in output_map:
        t = tf.cast(model.outputs[output_idx[t]], tf.float32)
        s = tf.cast(model.outputs[output_idx[s]], tf.float32)
        model.add_loss(tf.reduce_mean(tf.keras.losses.mean_squared_error(t, s)*scale))

    return model
