import os
import json
import shutil
import time
import yaml
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
tf.random.set_seed(2)
import random
random.seed(1234)
import numpy as np
np.random.seed(1234)

from profile import measure

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection
from taskhandler import *

dir_ = "build_build_approx"
#model_file = os.path.join(dir_, "students/nongated_studentgood.h5")
model_file = os.path.join(dir_, "base.h5")
#pretrained = "temp/emackpt-46"
pretrained = "../partial/temp/emackpt-99"

from keras_flops import get_flops
model = tf.keras.models.load_model(model_file, custom_objects={"SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer})

config = "configs/visdrone.yaml"
with open(config, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(1)

config["task"] = build_config(config["task"])
config["task"]["steps_per_execution"] = config["task"]["num_examples_per_epoch"] // config["task"]["batch_size"]
config["task"]["steps_per_epoch"] = config["task"]["num_examples_per_epoch"] // config["task"]["batch_size"]

model = post_prep_(config["task"], model, pretrained=pretrained, with_head=True)

#print(model.summary())
#from efficientnet.tfkeras import EfficientNetB2
#model = EfficientNetB2(weights='imagenet', input_shape=(260, 260, 3))

#from tensorflow.keras.applications import EfficientNetB0
#model = tf.keras.applications.efficientnet.EfficientNetB0(
#    include_top=True, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=1000,
#    classifier_activation='softmax')

#print(get_flops(model))

# remove front
flag = False
for layer in model.layers:
    if layer.name == "stem_conv_pad":
        flag = True
        break
if flag:
    removal = tf.keras.Model(model.input, model.get_layer("stem_conv_pad").output)
    removal_cpu = measure(removal, mode="onnx_cpu")
    removal_gpu = measure(removal, mode="onnx_gpu")
    print(removal_cpu)
else:
    removal_cpu = 0
    removal_gpu = 0

input_shape = [1, config["task"]["image_size"][0], config["task"]["image_size"][1], 3]
#print(model.summary())

#print(measure(model, mode="trt"))
#print(measure(model, mode="gpu"))
#print(measure(model, mode="gpu", batch_size=1))
#print(measure(model, mode="cpu"))
#print(measure(model, mode="tflite", batch_size=1))
print(measure(model, mode="onnx_cpu", input_shape=input_shape) - removal_cpu)
#print(measure(model, mode="onnx_gpu") - removal_gpu)
print(model.count_params())
