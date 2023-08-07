""" Alternative Set Topology

"""

from __future__ import absolute_import
from __future__ import print_function

from bespoke import backend as B

class Node(object):
    """This is a class of representing hierarchical alternatives of a model.

    """

    def __init__(self, id_, tag, net=None, pos=None, needs_finetune=True, is_original=False):
        """ Init

        """
        self._net = net
        self._neighbors = []
        self._id = id_
        self._profile = {}
        self._pos = pos
        self._needs_finetune = needs_finetune
        self._origin = None
        self._tag = tag
        self._is_original = is_original

    def add(self, new, w=0.0):
        """This function adds a neighbor into its neighbors list.

        """
        found = False
        for n, w in self._neighbors:
            if new.id_ == n.id_:
                found = True
        if not found:
            self._neighbors.append((new, w))

    def remove(self, target_id):
        """ Remove by target_id

        """
        found = None
        for c  in self._neighbors:
            if c[0].id_ == target_id:
                found = c
                break
        if found is not None:
            self._neighbors.remove(found)
        else:
            raise ValueError('%s is not included in the node set.' % target_id)

    def is_original(self):
        """ Determine whether it comes from the original model

        """
        return self._is_original

    @property
    def origin(self):
        """ What is its origin module

        """
        return self._origin

    @origin.setter
    def origin(self, origin):
        """ Setter for origin

        """
        self._origin = origin

    @property
    def tag(self):
        """ Tag naming

        """
        return self._tag

    @property
    def id_(self):
        """ Identifier

        """
        return self._id

    @property
    def neighbors(self):
        """ Neighbor nodes in the model

        """
        return [n for n in self._neighbors]

    @property
    def net(self):
        """ Backend network

        """
        return self._net
    
    @net.setter
    def net(self, net):
        """ Backend network setter

        """
        raise ValueError("Not allowed to set the net.")

    @property
    def pos(self):
        """ Position (the location of its origin node in the original model)

        """
        return self._pos
    
    @pos.setter
    def pos(self, pos):
        """ Setter for the origin location

        """
        raise ValueError("Not allowed to set the position.")

    def predict(self, data):
        """ Caller for predict function

        """
        return self._net.predict(data)

    def profile(self, sample_inputs, sample_outputs):
        """ Base Profile

        """
        if "app" in self.tag:
            temp = self._net.profile(sample_inputs, sample_outputs, cmodel=self.get_cmodel())
        else:
            temp = self._net.profile(sample_inputs, sample_outputs)
        self._profile.update(temp)

    def load_model(self, load_dir, custom_objects=None):
        """ Load model

        """
        model = B.load_model_from_node(load_dir, self.id_, custom_objects)
        self._net.model = model

    def sleep(self):
        """ Sleep function 

        """
        self._net.sleep()

    def wakeup(self):
        """ Weakup function

        """
        self._net.wakeup()

    def get_cmodel(self):
        """ Get nongated model

        """
        if self._origin.net.is_sleeping():
            self._origin.wakeup()
        origin_model = self._origin.net.model
        self._origin.sleep()
        return self.net.get_cmodel(origin_model)

    def save_model(self, save_dir):
        """ Save

        """
        return self._net.save(self.id_, save_dir)

    def serialize(self):
        """ Serialization

        """
        ret = {
            "id":self.id_,
            "tag":self._tag,
            "neighbors":[(n.id_, w) for n, w in self.neighbors],
            "pos":self._pos,
            "profile":self._profile,
            "origin":self._origin.id_ if self._origin is not None else "none",
            "is_original":self.is_original(),
            "meta":self.net.meta
        }
        return ret

    def load(self, dict_, nodes, custom_objects=None):
        """ Deserialization

        """
        model = B.load_model(dict_["model_path"], custom_objects)
        self._net = B.backend_net()(model, custom_objects)
        self._pos = [tuple(dict_["pos"][0]), tuple(dict_["pos"][1])]
        self._profile = dict_["profile"]
        for neighbor_id, w in dict_["neighbors"]:
            self.add(nodes[neighbor_id], w)
        if dict_["origin"] != "none":
            self._origin = nodes[dict_["origin"]]
        self._is_original = dict_["is_original"]
        self.net.meta = dict_["meta"]
        self._tag = dict_["tag"]
