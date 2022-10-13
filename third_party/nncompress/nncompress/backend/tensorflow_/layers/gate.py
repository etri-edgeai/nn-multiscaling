""" Pruning Gate Layer for TensorFlow """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from nncompress.assets.formula.gate import SimplePruningGateFormula

class SimplePruningGate(layers.Layer, SimplePruningGateFormula):
    """ Simple Pruning Gate Layer """

    def __init__(self,
                 ngates,
                 **kwargs):
        super(SimplePruningGate, self).__init__(**kwargs)
        self.ngates = ngates

        self.grad_holder = []
        self.collecting = True
        self.data_collecting = False

        @tf.custom_gradient
        def grad_tracker(x):
            def custom_grad(dy, variables):
                if self.collecting:
                    self.grad_holder.append(
                        tf.reduce_sum(dy * x, axis=[1, 2]))
                return self.compute(dy), [ tf.zeros(self.ngates,) ]
            return self.compute(x), custom_grad
        self.grad_tracker = grad_tracker

        self.data_holder = []
        def data_tracker(x):
            if self.data_collecting:
                self.data_holder.append(x)
            return x
        self.data_tracker = data_tracker

    def build(self, input_shape):
        """ Build function """
        self.gates = self.add_weight(name='gates',
                                     shape=(self.ngates,),
                                     initializer="ones",
                                     trainable=True)
        super(SimplePruningGate, self).build(input_shape)

    def call(self, input):
        """ Call function (grad tracking, data tracking, and applying gates) """
        return self.grad_tracker(self.data_tracker(input)), self.binary_selection()

    def compute_output_shape(self, input_shape):
        """ Compute the output shape of this module """
        return (input_shape[0], self.ngates)

    def get_config(self):
        """ Get config dict """
        config = super(SimplePruningGate, self).get_config()
        config.update({
            "ngates":self.ngates
        })
        return config
