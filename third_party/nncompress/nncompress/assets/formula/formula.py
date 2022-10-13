""" Formula. A complex operation holder. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod

from nncompress import backend as M

class Formula(ABC):
    """ Formula abstraction """

    @classmethod
    def instantiate(cls, postfix="", *args, **kwargs):
        """Instantiate an associated layer according to your backend.

        # Arguments
            *args: an args tuple.
            **kwargs: a kwargs dictionary.

        # Returns 
            An layer(module) instance on the backend.

        """
        # pylint: disable=E1102
        return M.get_type(cls.__name__[0:-7]+postfix)(*args, **kwargs)
        # pylint: enable=E1102

    @abstractmethod
    def compute(self, *input, training=False):
        """Compute the result for the given input.

        # Arguments
            input: input(Backend-aware Tensor).
            training: bool, a flag for identifying training.

        # Returns
            Result Tensor(s).

        """
