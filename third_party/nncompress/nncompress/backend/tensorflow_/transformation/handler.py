""" Handling classes """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
from tensorflow.keras.layers import Lambda, Concatenate

def get_handler(class_name):
    """ Retrieve the handler class defined by `class_name` """
    if class_name in LAYER_HANDLERS:
        return LAYER_HANDLERS[class_name]
    else:
        return LayerHandler

def cut(w, in_gate, out_gate):
    """ Cut a weight tensor with gate info """
    if out_gate is not None:
        out_gate = np.array(out_gate, dtype=np.bool)
        if len(w.shape) == 4: # conv2d
            w = w[:,:,:,out_gate]
        elif len(w.shape) == 2: # fc
            w = w[:,out_gate]
        elif len(w.shape) == 1: # bias ... 
            w = w[out_gate]
    if in_gate is not None:
        in_gate = np.array(in_gate, dtype=np.bool)
        if len(w.shape) == 4: # conv2d
            w = w[:,:,in_gate,:]
        elif len(w.shape) == 2: # fc
            w = w[in_gate,:]
    return w

class LayerHandler(object):
    """ Abstraction for keras-layer handler. """

    def __init__(self):
        pass

    @staticmethod
    def is_transformer(tensor_idx):
        """ Check whether this layer is a transformer or not """
        return False

    @staticmethod
    def is_concat():
        """ Check whether this layer is a concatenation layer """
        return False

    @staticmethod
    def get_output_modifier(name):
        """ Returns the output modifier """
        return None

    @staticmethod
    def get_gate_modifier(name):
        """ Returns the gate modifier """
        return None

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Update the layer schema """
        return

    @staticmethod
    def update_gate(gates, input_shape):
        """ Update gate info """
        return None

    @staticmethod
    def cut_weights(W, in_gate, out_gate):
        """ Cut a weight tensor """
        ret = []
        for w in W:
            w_ = cut(copy.deepcopy(w), in_gate, out_gate)
            ret.append(w_)
        return ret

class Conv2DHandler(LayerHandler):
    """ Conv2D Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ This is a transformer. """
        return True

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Layer schema update with new weights. """
        layer_dict["config"]["filters"] = new_weights[0].shape[-1]
        return

class WeightedSumHandler(LayerHandler):
    """ WeightedSum Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ This is not a transformer. """
        return False

    @staticmethod
    def cut_weights(W, in_gate, out_gate):
        """ Cut a weight tensor """
        ret = []
        for w in W:
            ret.append(w)
        return ret

class DenseHandler(LayerHandler):
    """ Dense Hander """
    
    @staticmethod
    def is_transformer(tensor_idx):
        """ This is a transformer. """
        return True

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Update layer schema with new weights. """
        layer_dict["config"]["units"] = new_weights[0].shape[-1]
        return

class ShiftHandler(LayerHandler):
    """ Shift Handler. """

    @staticmethod
    def is_transformer(tensor_idx):
        """ This is not a transformer. """
        return False

    @staticmethod
    def get_output_modifier(name):
        """ Return output modifier for shifting """
        """
            x[0] -> data
            x[1] -> mask
        """
        return Lambda(lambda x: x[0] * x[1], name=name)

class DWConv2DHandler(LayerHandler):
    """ Depthwise Conv2D Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ This is not a transformer. (Not cause channel change) """
        return False

    @staticmethod
    def get_output_modifier(name):
        """ Return output modifier.

            x[0] -> data
            x[1] -> mask
        """
        return Lambda(lambda x: x[0] * x[1], name=name)

    @staticmethod
    def cut_weights(W, in_gate, out_gate):
        """ Cut a weight tensor """
        ret = []
        for w in W:
            if len(w.shape) == 4:
                w_ = cut(copy.deepcopy(w), out_gate, None)
            else:
                w_ = cut(copy.deepcopy(w), in_gate, out_gate)
            ret.append(w_)
        return ret

class SeparableConv2DHandler(LayerHandler):
    """ SeparableConv2D Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ This is a transformer. (point-wise) """
        return True

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Update layer schema. """
        layer_dict["config"]["filters"] = new_weights[1].shape[-1]
        return

    @staticmethod
    def cut_weights(W, in_gate, out_gate):
        """ Cut a weight tensor """
        ret = []
        for idx, w in enumerate(W):
            if idx == 0: # Depth-wise
                w_ = cut(copy.deepcopy(w), in_gate, None)
            else: # Point-wise
                w_ = cut(copy.deepcopy(w), in_gate, out_gate)
            ret.append(w_)
        return ret

class ConcatHandler(LayerHandler):
    """ Concatenation Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ This is not a transformer. """
        return False

    @staticmethod
    def get_gate_modifier(name):
        """ Return gate modifier """
        return Concatenate(axis=-1, name=name)

    @staticmethod
    def is_concat():
        """ This is a concatenation layer. """
        return True

    @staticmethod
    def update_gate(gates, input_shape):
        """ Update gate """
        return np.concatenate(gates)

class FlattenHandler(LayerHandler):
    """ Flatten Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ This is not a transformer. """
        return False

    @staticmethod
    def update_gate(gates, input_shape):
        """ Update gate """
        shape = list(input_shape[1:]) # data_shape
        shape[-1] = 1 # not repeated at the last dim.
        return np.tile(gates, tuple(shape)).flatten()

class ReshapeHandler(LayerHandler):
    """ Reshape Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ This is not a transformer. """
        return False

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Update layer schema due to new weights. """
        val = int(np.sum(output_gate))
        layer_dict["config"]["target_shape"][-1] = val
        return

class InputLayerHandler(LayerHandler):
    """ InputLayer Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ This is a transformer. """
        return False

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Handling Input Layer Schema. """
        layer_dict["config"]["batch_input_shape"][-1] = int(np.sum(input_gate))
        return


LAYER_HANDLERS = {
    "Conv2D": Conv2DHandler,
    "Dense": DenseHandler,
    "BatchNormalization": ShiftHandler,
    "DepthwiseConv2D": DWConv2DHandler,
    "Concatenate": ConcatHandler,
    "Flatten": FlattenHandler,
    "Reshape": ReshapeHandler,
    "SeparableConv2D": SeparableConv2DHandler,
    "WeightedSum":WeightedSumHandler,
    "InputLayer":InputLayerHandler
}
