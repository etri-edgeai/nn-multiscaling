"""
    Bespoke Runner
"""

from __future__ import print_function
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()

import json
import tempfile
import os
import copy
import sys

from bespoke.base.interface import ModelHouse
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import StopGradientLayer

from butils import optimizer_factory

custom_objects = {
    "SimplePruningGate":SimplePruningGate,
    "StopGradientLayer":StopGradientLayer,
    "HvdMovingAverage":optimizer_factory.HvdMovingAverage
}

def run():

    src = sys.argv[1]
    dst = sys.argv[2]

    mh = ModelHouse(None, custom_objects=custom_objects)
    mh.load(src)
    mh.save(dst)

    print("source: ", src)
    print("dest: ", dst)

if __name__ == "__main__":
    run()
