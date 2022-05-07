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

print(tf.config.experimental.get_memory_info("GPU:0"))

mh = ModelHouse(None)
mh.load("saved_efnet_house_app")
for n in mh.nodes:
   n.sleep() # to_cpu 

train_data_generator, _, _ = load_data(dataset, model_handler, training_augment=True, n_classes=100)
sample_inputs = []
for x,y in train_data_generator:
    sample_inputs.append(x)
    if len(sample_inputs) > 30:
        break

#mh = ModelHouse(model=None)
#mh.load("ttt")
mh.build_sample_data(sample_inputs)
mh.profile()

mh.save("saved_efnet_house_profile")

# check mh and mh2 are same.
#assert mh._namespace == mh2._namespace
#for n1, n2 in zip(mh._nodes, mh2._nodes):
#    assert (n1.net.model.get_weights()[0] == n2.net.model.get_weights()[0]).all()


#data = np.random.rand(1,224,224,3)
#y = house(data)
#tf.keras.utils.plot_model(house, to_file="house.pdf", show_shapes=True)


#iteration_based_train(dataset, house, model_handler, 500, output_idx, output_map, lr_mode=0, stopping_callback=None, augment=True, n_classes=100, eval_steps=-1, validate_func=None)
#mh.extract({5: 0, 1: 1})
