from __future__ import absolute_import
from __future__ import print_function

class Node(object):
    """This is a class of representing hierarchical alternatives of a model.

    """

    def __init__(self, id_, net):
        self._net = net
        self._children = []
        self._parent = None
        self._id = id_
        self._profile = {}

    def add(self, id_, net, pos=None):
        """This function adds a child node into its children list.

        """
        if pos is None:
            child = AlternativeNode(id_, net)
            e = Edge(child)
        else:
            child = BranchingNode(id_, net)
            e = PositionEdge(child, pos)
        self._children.append(e)
        child.parent = e

    def remove(self, target_id):
        found = None
        for c in self._children:
            if c.node.id_ == target_id:
                found = c
                break
        if found is not None:
            self._children.remove(found)
            found.node.parent = None
        else:
            raise ValueError('%s is not included in the node set.' % target_id)

    @property
    def id_(self):
        return self._id

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, e):
        self._parent = e

    @property
    def children(self):
        return [child for child in self._children]

    @property
    def net(self):
        return self._net
    
    @net.setter
    def net(self, net):
        raise ValueError("Not allowed to set the net.")

    @property
    def net(self):
        return self._net

    def predict(self, data):
        return self._net.predict(data)

    def set_profile(self, profiler, recursive=True):
        self._profile = profiler(self._net)
        if recursive:
            for child in self._children:
                child.node.set_profile(profiler, recursive)


class BranchingNode(Node):

    def __init__(self, id_, net):
        super(BranchingNode, self).__init__(id_, net)


class AlternativeNode(Node):

    def __init__(self, id_, net):
        super(AlternativeNode, self).__init__(id_, net)


class Edge(object):
    """This is a class of representing the position where an alternative is located in.
    """

    def __init__(self, node):
        self._node = node

    @property
    def node(self):
        return self._node


class PositionEdge(Edge):
    """This is a class of representing the position where an alternative is located in.
    """

    def __init__(self, node, pos):
        super(PositionEdge, self).__init__(node)
        self._pos = pos

    @property
    def pos(self):
        return self._pos
