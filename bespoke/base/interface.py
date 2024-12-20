""" ModelHouse Implementation

"""

from __future__ import absolute_import
from __future__ import print_function

import json
import pickle
import os
import copy

import numpy as np

from bespoke import backend as B
from bespoke.base.topology import Node
from bespoke.generator import *

from nncompress.algorithms.solver.solver import State
from nncompress.algorithms.solver.simulated_annealing import SimulatedAnnealingSolver

class ModelState(State):
    """ A state for executing the simulated annealing solver

    """

    def __init__(self, selected_nodes, nodes, parser, metric):
        self.selected_nodes = selected_nodes
        self.nodes = nodes
        self.parser = parser
        self.metric = metric
            
    def get_next_impl(self):
        c = copy.copy(self.selected_nodes)
        while len(c) > 0:
            random.shuffle(c)
            c.pop()

            if random.random() > 0.25:
                break

        while True:
            comple = []
            for n in self.nodes:
                if n._profile[self.metric] >= 1000000.0 and self.metric != "flops":
                    continue
                on = n.origin
                if on is not None:
                    while on.origin != None:
                        on = on.origin
                if on is not None:
                    if on._profile[self.metric] >= 1000000.0 and self.metric != "flops":
                        continue

                compatible = True
                for c_ in c:
                    if not self.parser.is_compatible(n, c_):
                        compatible = False
                        break
                if compatible:
                    comple.append(n)

            if len(comple) == 0:
                break
            else:
                n = np.random.choice(comple)
                c.append(n)
           
        return ModelState(c, self.nodes, self.parser, self.metric)


def score_f(obj_value, base_value, metric, lda, nodes):
    """ Score function """
    approx_value = base_value
    score = None
    sum_mse = 0
    sum_iacc = 0
    for n in nodes:
        on = n.origin
        if on is not None:
            while on.origin != None:
                on = on.origin
        if on is not None:
            if metric != "igpu":
                approx_value = approx_value - on._profile[metric] + n._profile[metric]
            else:
                approx_value = approx_value - n._profile[metric]
            sum_iacc += on._profile["iacc"] - n._profile["iacc"]
        sum_mse += n._profile["mse"]

    score = max(approx_value / obj_value, 1.0) + sum_iacc * lda
    print(approx_value, obj_value, sum_iacc, score)
    return -1 * score


class ModelHouse(object):
    """This is a class of housing a model for model scaling.

    """

    def __init__(self, model=None, custom_objects=None):
        if model is None:
            self._model = model
            self._parser = None
            self._custom_objects = custom_objects
            self._nodes = None
            self._namespace = None
        else:
            self._namespace = set()
            model_, parser = B.preprocess(model, self._namespace, custom_objects)
            self._model = model_
            self._parser = parser
            self._custom_objects = custom_objects
            self._nodes = []
        self._sample_inputs = None
        self._sample_outputs = None

    def build_base(
        self,
        model_list=None,
        min_num=20,
        memory_limit=None,
        params_limit=None,
        step_ratio=0.1,
        use_last_types=False):
        """ Build Base """
        if model_list is not None and len(model_list) == 0:
            return

        nodes_ = []
        gen_ = PretrainedModelGenerator(self._namespace, model_list=model_list)
        while len(nodes_) < min_num:
            print(len(nodes_))
            n = np.random.choice(self._nodes)
            if use_last_types:
                n.wakeup()
                _, parser = B.preprocess(n.net.model, set(), self._custom_objects) # dummy namespace
                last_types = parser.get_last_types()
                n.sleep()
                alters = gen_.generate(
                    n.net,
                    last_types,
                    memory_limit=memory_limit,
                    params_limit=params_limit,
                    step_ratio=step_ratio,
                    use_adapter=True)
            else:
                alters = gen_.generate(
                    n.net,
                    n.pos[1][0],
                    memory_limit=memory_limit,
                    params_limit=params_limit,
                    step_ratio=step_ratio,
                    use_adapter=True)
            for idx, (a, model_name) in enumerate(alters): 
                na = Node(self._parser.get_id("anode"), "alter_"+model_name, a, pos=n.pos)
                na.origin = n
                nodes_.append(na)
                na.sleep()
        self._nodes.extend(nodes_)

    def build_approx(
        self,
        min_num=20,
        memory_limit=None,
        params_limit=None,
        init=False,
        pruning_exit=False,
        data=None):
        """ Build Approximation """
        if data is not None:
            self.build_sample_data(data)

        gen_ = PruningGenerator(self._namespace)
        nodes_ = []
        while len(nodes_) < min_num:
            print(len(nodes_))
            n = np.random.choice(self._nodes)
            tag = "app_origin" if n.tag == "origin" else "app_alter"
            scale = np.random.choice([0.05, 0.075, 0.1, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75])
            sample_data = self._sample_inputs[n.pos[0]]
            alters = gen_.generate(
                n.net,
                [scale],
                sample_data=sample_data,
                custom_objects=self._custom_objects,
                init=init,
                pruning_exit=pruning_exit)

            if not alters:
                continue

            for idx, a in enumerate(alters):
                nodes_.append(Node(self._parser.get_id("anode"), tag, a, pos=n.pos))
                nodes_[-1].origin = n
        self._nodes.extend(nodes_)

    def get_node(self, id_):
        """ Get node by id_ """
        for n in self._nodes:
            if n.id_ == id_:
                return n
        return None

    def select(self, spec, return_gated_model=False, lda=0.1, ratio=1.0, use_random=False):
        """ Select function """

        if use_random:
            print("RANDOM SELECTION")
            return self._rand_select(spec, return_gated_model, lda, ratio)
        else:
            return self._select(spec, return_gated_model, lda, ratio)


    def _rand_select(self, spec, return_gated_model=False, lda=0.1, ratio=1.0):
        """ Private random """
        metric, obj_value, base_value = spec
        approx_value = base_value
       
        COUNT = 10000
        nodes = [n for n in self._nodes if n.is_original()]
        import random
        minimal = []
        for i in range(COUNT):
            random.shuffle(nodes)
            node = nodes[0]

            anodes = []
            for node_ in self._nodes:
                if not node_.is_original():
                    on = node_.origin
                    while on.origin != None:
                        on = on.origin

                    if node == on:
                        anodes.append(node_)
            
            random.shuffle(anodes)

            for a in anodes:

                compatible = True
                for m in minimal:
                    if not self._parser.is_compatible(a, m):
                        compatible = False
                        break
                if not compatible:
                    break

                if a not in minimal:
                    minimal.append(a)
                    break

        for m in minimal:
            print(m.id_, m.tag, m.pos)

        ret = self._parser.extract(self.origin_nodes, minimal, return_gated_model=return_gated_model)
        return ret

    def _select(self, spec, return_gated_model=False, lda=0.1, ratio=1.0):
        """ SA selection """
        minimal = []
        nodes = [n for n in self._nodes]
        import random
        random.shuffle(nodes)
        if ratio < 1.0:
            nodes = nodes[:int(len(nodes)*ratio)]

        num_iters = 100
        iter_ = 0
        metric, obj_value, base_value = spec
        approx_value = base_value

        print(metric, base_value)
        while True:
            min_ = -1
            min_n = None
            min_on = None
            old_len = len(minimal)
            for n in nodes:
                if n._profile[metric] >= 1000000.0 and metric != "flops":
                    continue

                on = n.origin
                if on is not None:
                    while on.origin != None:
                        on = on.origin

                if on is not None and on._profile[metric] >= 1000000.0 and metric != "flops":
                    continue

                compatible = True
                for m in minimal:
                    if not self._parser.is_compatible(n, m):
                        compatible = False
                        break
                if compatible:
                    if on is None:
                        score = max(approx_value / obj_value, 1.0)
                    else:
                        if metric != "igpu":
                            score = max((approx_value - on._profile[metric] + n._profile[metric]) / obj_value, 1.0) +\
                                (on._profile["iacc"] - n._profile["iacc"]) * lda
                        else:
                            score = max((approx_value - n._profile[metric]) / obj_value, 1.0) + (on._profile["iacc"]
                                - n._profile["iacc"]) * lda

                    if min_== -1 or min_ > score:
                        min_ = score
                        min_on = on
                        min_n = n

            if min_n is not None and min_n not in minimal:
                minimal.append(min_n)
                if min_on is not None:
                    if metric != "igpu":
                        approx_value = approx_value - min_on._profile[metric] + min_n._profile[metric]
                    else:
                        approx_value = approx_value - min_n._profile[metric]

            if old_len == len(minimal) or iter_ >= num_iters: # not changed
                break
            else:
                iter_ += 1

        score_func = lambda state: score_f(obj_value, base_value, metric, lda, state.selected_nodes)
        sa = SimulatedAnnealingSolver(score_func, max_niters=5000)
        init_state = ModelState(minimal, nodes, self._parser, metric)
        last, best = sa.solve(init_state)
        minimal = best.selected_nodes

        for m in minimal:
            print(m.id_, m.tag, m.pos)

        ret = self._parser.extract(self.origin_nodes, minimal, return_gated_model=return_gated_model)
        return ret


    @property
    def origin_nodes(self):
        """ Origin nodes """
        ret = {}
        for n in self._nodes:
            if n.is_original():
                ret[tuple(n.pos)] = n
        return ret

    @property
    def model(self):
        """ Getter for model """
        return self._model

    @property
    def parser(self):
        """ Getter for parser """
        return self._parser

    @property
    def nodes(self):
        """ Getter for nodes """
        return self._nodes

    @property
    def trainable_nodes(self):
        """ Get trainable nodes """
        nodes = []
        for n in self._nodes:
            if not n.is_original():
                nodes.append(n)
        return nodes

    def extend(self, nodes):
        """ Extends node set """
        self._nodes += nodes

    def build_sample_data(self, data):
        """ Build sample data for profiling """
        self._sample_inputs = {}
        self._sample_outputs = {}
        for n in self._nodes:
            if n.is_original():
                ret = B.build_samples(self._model, data, n.pos)
                self._sample_inputs[n.pos[0]] = [ret_[0] for ret_ in ret]
                self._sample_outputs[n.pos[1]] = [ret_[1] for ret_ in ret]

    def make_train_model(self, nodes, scale=0.1):
        """ Prepare a model for training """
        return B.make_train_model(self._model, nodes, scale=scale)

    def profile(self):
        """ Base Profile """
        if self._sample_inputs is None:
            raise ValueError("build_sample_data should've been called before profiling.")

        for n in self._nodes:
            n.profile(self._sample_inputs[n.pos[0]], self._sample_outputs[n.pos[1]])

    def _get_predefined_paths(self, dir_):
        """ Get some predefined paths """
        subnet_dir_path = os.path.join(dir_, "nets")
        nodes_path = os.path.join(dir_, "nodes.json")
        namespace_path = os.path.join(dir_, "namespace.pkl")

        return subnet_dir_path,\
            nodes_path,\
            namespace_path

    def save(self, save_dir):
        """ Save model house """
       
        if os.path.exists(save_dir):
            print("%s exists." % save_dir)
        else:
            os.mkdir(save_dir)

        subnet_dir, nodes_path, namespace_path = self._get_predefined_paths(save_dir)
       
        # construct predefined dirs
        if os.path.exists(subnet_dir):
            print("%s exists." % subnet_dir)
        else:
            os.mkdir(subnet_dir)

        serialized = {
            node.id_:node.serialize()
            for node in self._nodes
        }
        for node in self._nodes:
            path = node.save_model(subnet_dir)
            serialized[node.id_]["model_path"] = path

        with open(nodes_path, "w") as f:
            json.dump(serialized, f, indent=4)

        B.save_model("base", self._model, save_dir)
        with open(namespace_path, "wb") as f:
            pickle.dump(self._namespace, f)

    def load(self, load_dir):
        """ Load from a directory """
        subnet_dir, nodes_path, namespace_path = self._get_predefined_paths(load_dir)
        with open(nodes_path, "r") as f:
            serialized = json.load(f)

            self._nodes = []
            node_dict = {}
            for key, node_s in serialized.items():
                self._nodes.append(Node(key, None)) # empty node with identifier
                node_dict[key] = self._nodes[-1]
                
            for key, node_s in serialized.items():
                node_dict[key].load(node_s, node_dict, self._custom_objects) 

        model_path = B.get_basemodel_path(load_dir)
        self._model = B.load_model(model_path, self._custom_objects)

        with open(namespace_path, "rb") as f:
            self._namespace = pickle.load(f)

        self._parser = B.get_parser(self._model, self._namespace, self._custom_objects)
        print("load!")

    def add(self, node):
        """ Add node """
        self._nodes.append(node)

    def remove(self, node):
        """ Remove node """
        self._nodes.remove(node)
