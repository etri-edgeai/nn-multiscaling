from __future__ import absolute_import
from __future__ import print_function

import os
import random
import json
import copy

from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from keras_flops import get_flops

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation import parse, inject, cut, unfold
from nncompress.backend import add_prefix

from bespoke.backend import common
from bespoke.base.topology import Node

MAX_TRY = 20

def preprocess(model, namespace, custom_objects):
    model = unfold(model, custom_objects)
    return model, TFParser(model, namespace, custom_objects)

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
            if layer.__class__.__name__ in ["Conv2D", "Dense", "DepthwiseConv2D"]:
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

        if irank_a > orank_b:
            return True
        if orank_a < irank_b:
            return True
        return False

    def get_random_subnets(self, num=1, target_shapes=None, memory_limit=None, params_limit=None):
        nets = []
        for i in range(num):
            layers_ = []
            num_try = 0
            target_shapes_ = None
            subnet = None
            while True:
                if num_try > MAX_TRY:
                    return None
                num_try += 1
                #print(self._joints)

                # select two positions from the joints randomly
                left_idx = random.randint(0, len(self._joints)-5)
                step = int(len(self._joints) * 0.2)
                min_ = min(len(self._joints)-1, left_idx + step)
                right_idx = random.randint(left_idx+1, min_)
                left = self._trank[self._joints[left_idx]]
                right = self._trank[self._joints[right_idx]]

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
                        input_shape[-1] <= left_shape[-1] and output_shape[-1] <= right_shape[-1] and\
                        channel_change == target_cchange):
                        continue
                    #else:
                    #    print("success!")
                    #    print(self._model.get_layer(self._rtrank[left]).name, self._model.get_layer(self._rtrank[right]).name)
                    #    print(target_shapes)
                    #    print(left_shape, right_shape)

                layers_ = [
                    self._model.get_layer(self._rtrank[j])
                    for j in range(left, right+1)
                ]

                if not self._is_compressible(layers_):
                    continue

                # Assumption: # of inputs is 1.
                target_shapes_ = [target_shapes[0]] if target_shapes is not None else None
                subnet = self._parser.get_subnet(layers_, self._model, target_shapes_)

                if memory_limit is not None:
                    shape = list(subnet[0].input.shape)
                    # TODO 32->other number
                    shape[0] = 32
                    data = np.random.rand(*shape)
                    tf.config.experimental.reset_memory_stats('GPU:0')
                    peak1 = tf.config.experimental.get_memory_info('GPU:0')['peak']
                    subnet[0](data)
                    peak2 = tf.config.experimental.get_memory_info('GPU:0')['peak']
                    if memory_limit < peak2-peak1:
                        continue
                    break
                if params_limit is not None:
                    if subnet[0].count_params() > params_limit:
                        continue
                else:
                    break

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
                    #print(input_mask.shape)
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

                    print(lgtop)
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
                                #print(key, w.shape, sum_.shape, dim)
                                for i in range(dim):
                                    for v in val:
                                        if v[0] <= i and i < v[1]:
                                            j = i - v[0]
                                            #print(i, j, v[0], v[1], len(score), sum_.shape)
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
                            print(g_)
                            output_gate = subnet_gmodel.get_layer(gm[(g_, 0)][0]["config"]["name"])
                            print(output_gate.name)
                            output_gate.gates.assign(output_mask)
                            print(output_gate)

                new_spatial_shape = {
                    subnet[0].input.name: input_shape
                }
                subnet_cmodel = subnet_parser.cut(subnet_gmodel, new_spatial_shape=new_spatial_shape)
                assert subnet_cmodel.input.shape[-1] == target_shapes[0][-1]
                del subnet_gmodel

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
            return isinstance(self._model.get_layer(node_dict["layer_dict"]["name"]), tf.keras.layers.Activation) or\
                isinstance(self._model.get_layer(node_dict["layer_dict"]["name"]), tf.keras.layers.ReLU)
            #return len(node_dict["layer_dict"]["inbound_nodes"]) > 0 and len(node_dict["layer_dict"]["inbound_nodes"][0]) == 1
        joints = self._parser.get_joints(filter_=f)

        """
        groups = self._parser.get_sharing_groups()
        groups_ = []
        r_groups = {}
        for group in groups:
            group_ = []
            for layer_name in group:
                group_.append(self._model.get_layer(layer_name))
                r_groups[layer_name] = group_
            groups_.append(group_)

        self._groups = groups_
        self._r_groups = r_groups
        """
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

    def profile(self, sample_inputs, sample_outputs):
        return {
            "flops": self.get_flops(),
            "mse": float(self.get_mse(sample_inputs, sample_outputs).numpy())
        }

    def get_flops(self):
        if self.is_sleeping():
            self.wakeup()
        flops = get_flops(self._model, batch_size=1)
        return flops

    def get_mse(self, sample_inputs, sample_outputs):
        if self.is_sleeping():
            self.wakeup()
        outputs = self._model(sample_inputs)
        return tf.reduce_mean(tf.keras.losses.mean_squared_error(sample_outputs, outputs))

    def get_cmodel(self, origin_model):
        self.wakeup()
        flag = False
        for layer in self._model.layers:
            if layer.__class__ == SimplePruningGate:
                flag = True
                print("xxx")
                print(np.sum(layer.gates))
                print(layer.ngates)
                print("xxx")
                break
        if not flag:
            return self.model

        tf.keras.utils.plot_model(self._model, "testing.pdf", show_shapes=True)
        parser = PruningNNParser(origin_model, allow_input_pruning=True, custom_objects=self._custom_objects, gate_class=SimplePruningGate)
        parser.parse()
        cmodel = parser.cut(self._model)
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


def extract(parser, origin_nodes, nodes):

    gmodel, gm = parser.inject(with_mapping=True, with_splits=True)
    for layer in gmodel.layers: 
        if layer.__class__ == SimplePruningGate:
            layer.gates.assign(np.ones((layer.ngates,)))

    groups = parser.get_sharing_groups()
    r_groups = {}
    for group in groups:
        for layer_name in group:
            r_groups[layer_name] = group
    affecting = parser.get_affecting_layers()

    # get a pruned model with union consensus
    cparser = {}
    exit_gates = {}
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
            last = cparser[n.id_].get_last_transformers()
            if type(last) == str:
                last = [last]
            for t in last:
                gate_name = cgm[t][0]["config"]["name"]
                exit_gates[t] = n.net.model.get_layer(gate_name).gates.numpy()

                """
                affecting_layers = affecting[(output_[0], 0)]
                for g in r_groups[affecting_layers[0][0]]:
                    target_gate = gmodel.get_layer(gm[(g, 0)][0]["config"]["name"])
                    tgates = target_gate.gates.numpy()
                    union = (((gates + tgates) == 0.0) == False).astype(np.float32)
                    target_gate.gates.assign(union)
                    union_gates.add(target_gate.name)
                """

    print("here")
    #model = parser.cut(gmodel)
    #parser_ = PruningNNParser(model, custom_objects=parser._custom_objects) 
    #parser_.parse()
    replacing_mappings = []
    in_maps = None
    ex_maps = []
    cmodel_map = {}
    for idx, n in enumerate(nodes):
        print(idx)
        if n.origin is None:
            continue
        pos = n.pos
        input_, output_ = pos
        target_gate = gmodel.get_layer(gm[(output_[0], 0)][0]["config"]["name"])

        # ones assigning
        backup = {}
        last = cparser[n.id_].get_last_transformers()
        if type(last) == str:
            last = [last]
        cgroups = cparser[n.id_].get_sharing_groups()
   
        for cg in cgroups:
            if n.net.model.input.name in cg:
                for l in cg:
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
                        break
        
        cmodel = n.get_cmodel()
        cmodel_map[n.id_] = cmodel

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

        original_input = copy.deepcopy(parser.get_layer_dict(pos[0][0]))
        original_input["inbound_nodes"] = []
        replacement = [original_input]
        inputs_ = []
        for layer in json.loads(cmodel.to_json())["config"]["layers"]:
            if layer["class_name"] != "InputLayer": # remove input layer
                replacement.append(layer)
            else:
                inputs_.append(layer["config"]["name"])
        tf.keras.utils.plot_model(cmodel, "cmodel%d.pdf" % idx, show_shapes=True)
        for layer in replacement:
            for flow in layer["inbound_nodes"]:
                for inbound in flow:
                    if inbound[0] in inputs_:
                        inbound[0] = original_input["config"]["name"]
        ex_map = [
            [(pos[0][0], original_input["config"]["name"])],
            [(pos[-1][0], replacement[-1]["name"], 0, 0)]
        ]
        ex_maps.append(ex_map)
        replacing_mappings.append((target, replacement))

    # Conduct replace_block
    model_dict = parser.replace_block(replacing_mappings, in_maps, ex_maps, parser.custom_objects)
    model_json = json.dumps(model_dict) 
    model = tf.keras.models.model_from_json(model_json, custom_objects=parser.custom_objects)

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

    parser_ = PruningNNParser(model, custom_objects=parser._custom_objects, gate_class=SimplePruningGate) 
    parser_.parse()
    gmodel, gm = parser_.inject(with_splits=True, with_mapping=True)

    sharing_groups = parser_.get_sharing_groups()

    for t in exit_gates:
        gates = exit_gates[t]
        target_gate = gmodel.get_layer(gm[(t, 0)][0]["config"]["name"])
        tgates = target_gate.gates.numpy()
        union = (((gates + tgates) == 0.0) == False).astype(np.float32)
        target_gate.gates.assign(union)

    for t in exit_gates:
        gates = exit_gates[t]
        target_gate = gmodel.get_layer(gm[(t, 0)][0]["config"]["name"])
        tgates = target_gate.gates.numpy()
        for g in sharing_groups:
            if t in g:
                for l in g:
                    target_gate = gmodel.get_layer(gm[(l, 0)][0]["config"]["name"])
                    target_gate.gates.assign(tgates)
    
    ret = parser_.cut(gmodel)

    tf.keras.models.save_model(ret, "result.h5", overwrite=True)
    tf.keras.utils.plot_model(ret, "ret.pdf")
    print(ret.summary())
    xxx


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
            if l.__class__ == SimplePruningGate:
                l.collecting = False
                l.data_collecting = False

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
    for (t, s) in output_map:
        house.add_loss(tf.reduce_mean(tf.keras.losses.mean_squared_error(t, s)*scale))

    return house, output_idx, output_map


def backend_net():
    return TFNet


def get_basemodel_path(dir_):
    return os.path.join(dir_, "base.h5")
    

def prune(net, scale, namespace, custom_objects=None):

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
    parser = PruningNNParser(net.model, allow_input_pruning=True, custom_objects=custom_objects, gate_class=SimplePruningGate, namespace=namespace)
    parser.parse()

    gated_model, gm = parser.inject(with_splits=True, allow_pruning_last=True, with_mapping=True)
    for layer in gated_model.layers:
        if type(layer) == SimplePruningGate:
            layer.collecting = False
            layer.data_collecting = False
    for layer in gated_model.layers:
        if layer.__class__ == SimplePruningGate:
            layer.gates.assign(np.ones((layer.ngates,)))

    last_transformers = parser.get_last_transformers()
    target_groups, gate_struct = parser.get_group_topology()
    for idx, (g, dict_) in enumerate(zip(target_groups, gate_struct)):
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
        val = sorted_[int((len(sorted_)-1)*scale)]
        mask = (mask >= val).astype(np.float32)

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
    cutmodel = parser.cut(gated_model)

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
        return model

def generate_pretrained_models():
    whitelist = [
        #tf.keras.applications.ResNet50V2,
        tf.keras.applications.InceptionResNetV2,
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.MobileNet,
        #tf.keras.applications.DenseNet121,
        #tf.keras.applications.NASNetMobile,
        tf.keras.applications.EfficientNetB1,
        tf.keras.applications.EfficientNetV2B1,
        tf.keras.applications.EfficientNetV2S
    ]
    models = []
    for w in whitelist:
        models.append(TFParser(w(include_top=True, weights="imagenet", classes=1000), namespace=set()))
    return models

def build_samples(model, data_gen, pos):
    outputs = [[],[]]
    for idx, p in enumerate(pos):
        for pp in p:
            outputs[idx].append(model.get_layer(pp).output)

    extractor = tf.keras.Model(model.inputs, outputs)
    for data in data_gen:
        results = extractor(data)
    return results