import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from bespoke.base.interface import ModelHouse

import numpy as np
from tensorflow import keras
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)
"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import resnet50 as model_handler
from train import train, iteration_based_train

dataset = "cifar100"


#model = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling=None, classes=10)
#model = tf.keras.applications.DenseNet121(include_top=False, weights=None, pooling=None, classes=10)

model = model_handler.get_model("cifar100")

#model_handler.compile(model, run_eagerly=True, loss="categorical_crossentropy")
#train(dataset, model, "test", model_handler, 5, callbacks=None, augment=True, exclude_val=False, n_classes=100)

tf.keras.utils.plot_model(model, to_file="original.png", show_shapes=True)

mh = ModelHouse(model)

house, output_idx, output_map = mh.make_train_model(range_=[3,4])

#print(house.summary())

data = np.random.rand(1,224,224,3)
y = house(data)

tf.keras.utils.plot_model(house, to_file="house.pdf", show_shapes=True)

model_handler.compile(house, run_eagerly=True)

print([ l.name for l in house.trainable_variables])

train(dataset, house, "test", model_handler, 5, callbacks=None, augment=True, n_classes=100)
#iteration_based_train(dataset, house, model_handler, 500, output_idx, output_map, lr_mode=0, stopping_callback=None, augment=True, n_classes=100, eval_steps=-1, validate_func=None)
#mh.extract({5: 0, 1: 1})
