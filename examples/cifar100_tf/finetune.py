import os
import copy
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from orderedset import OrderedSet

from nncompress.backend.tensorflow_.transformation import handler
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


model = tf.keras.models.load_model("result.h5")
model_handler.compile(model, run_eagerly=True, loss="categorical_crossentropy")
train(dataset, model, "test", model_handler, 10, callbacks=None, augment=True, exclude_val=False, n_classes=100)
