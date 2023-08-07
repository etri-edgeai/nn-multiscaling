""" Profiler Wrapper

"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from bespoke import backend as B

class Profiler(object):
    """ Profiler class

    """

    def __init__(self):
        """ Init function """
        pass

    def profile(self, target):
        """ Profile function

        """
        return B.profile(target)
