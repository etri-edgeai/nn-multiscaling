from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from bespoke import backend as B

class Profiler(object):

    def __init__(self):
        pass

    def profile(self, target):
        return B.profile(target)
