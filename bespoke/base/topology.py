""" Topology """

from __future__ import absolute_import
from __future__ import print_function

from bespoke import backend as B

class Node(object):
    """This is a class of representing hierarchical alternatives of a model.

    """

    def __init__(self, id_, tag, net=None, pos=None, needs_finetune=True, is_original=False):
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
        """ This function remove a neighbor form its neighbor list.

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
        """ This function returns whether it is original or not.

        """
        return self._is_original

    @property
    def origin(self):
        """ This function returns where this node comes.

        """
        return self._origin

    @origin.setter
    def origin(self, origin):
        self._origin = origin

    @property
    def tag(self):
        """ This function returns tag info.

        """
        return self._tag

    @property
    def id_(self):
        """ Identifier

        """
        return self._id

    @property
    def neighbors(self):
        """ Neighbors

        """
        return [n for n in self._neighbors]

    @property
    def net(self):
        """ Return associated subnetwork.

        """
        return self._net
    
    @net.setter
    def net(self, net):
        raise ValueError("Not allowed to set the net.")

    @property
    def pos(self):
        """ Related position in the original network.

        """
        return self._pos
    
    @pos.setter
    def pos(self, pos):
        raise ValueError("Not allowed to set the position.")

    def predict(self, data):
        """ Wrapper function for `predict`.

        """
        return self._net.predict(data)

    def profile(self, sample_inputs, sample_outputs):
        """ Profile function.

        """
        if "app" in self.tag:
            temp = self._net.profile(sample_inputs, sample_outputs, cmodel=self.get_cmodel())
        else:
            temp = self._net.profile(sample_inputs, sample_outputs)
        self._profile.update(temp)

    def load_model(self, load_dir, custom_objects=None):
        """ Load its model from `load_dir`.

        """
        model = B.load_model_from_node(load_dir, self.id_, custom_objects)
        self._net.model = model

    def sleep(self):
        """ GPU -> CPU

        """
        self._net.sleep()

    def wakeup(self):
        """ CPU -> GPU

        """
        self._net.wakeup()

    def get_cmodel(self):
        """ This function returns a compressed version if it has gates.

        """
        if self._origin.net.is_sleeping():
            self._origin.wakeup()
        origin_model = self._origin.net.model
        self._origin.sleep()
        return self.net.get_cmodel(origin_model)

    def save_model(self, save_dir):
        """ Save a model.

        """
        return self._net.save(self.id_, save_dir)

    def serialize(self):
        """ Dictionary-based serialization.

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
        """ Load from filesystem.

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
