from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nncompress.assets.formula.formula import Formula
from nncompress import backend as M

def b(x):
    # Assume that > operator must be supported in backends.
    return M.cast((x >= 0.5), "float32")

def gate_func(x, L=10e5, grad_shape_func=None):
    x_ = x - 0.5
    if callable(grad_shape_func):
        return b(x) + ((L * x_ - M.floor(L * x_)) / L) * grad_shape_func(x_)
    elif grad_shape_func is not None:
        return b(x) + ((L * x_ - M.floor(L * x_)) / L) * M.function(grad_shape_func, x_)
    else:
        return b(x) + ((L * x_ - M.floor(L * x_)) / L)

class SimplePruningGateFormula(Formula):

    def __init__(self):
        super(SimplePruningGateFormula, self).__init__()

    def compute(self, input):
        return M.cmul(input, self.binary_selection())

    def binary_selection(self):
        return M.round(self.gates)  # gates consists of ones or zeros.

    def get_sparsity(self):
        selection = self.binary_selection()
        return 1.0 - M.sum(selection) / self.gates.shape[0]
