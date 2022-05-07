import os
import copy
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from orderedset import OrderedSet

from nncompress.backend.tensorflow_.transformation import handler
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection
from bespoke.base.interface import ModelHouse
from bespoke.base.builder import RandomHouseBuilder

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#import resnet50 as model_handler
from models import efficientnet as model_handler
from train import train, iteration_based_train, load_data

dataset = "cifar100"

print(tf.config.experimental.get_memory_info("GPU:0"))

custom_objects = {
    "SimplePruningGate":SimplePruningGate,
    "StopGradientLayer":StopGradientLayer
}

mh = ModelHouse(None, custom_objects=custom_objects)
mh.load("saved_efnet_house_profile")
for n in mh.nodes:
   n.sleep() # to_cpu 

mh.select()
