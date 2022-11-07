""" Base profiling """

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
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
from nncompress.backend.tensorflow_.transformation.pruning_parser import StopGradientLayer
from nncompress.backend.tensorflow_.transformation.pruning_parser import has_intersection
from taskhandler import *
from prep import *

dir_ = "../search/build_build_approx"
#model_file = os.path.join(dir_, "students/nongated_studentgood.h5")
model_file = os.path.join(dir_, "base.h5")
pretrained = "../../experiments/partial/temp/emackpt-99"
#pretrained = "pretrained_weights/efficientdet-d0"

from keras_flops import get_flops
model = tf.keras.models.load_model(
    model_file, custom_objects={"SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer})

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

model = post_prep_(config["task"], model, pretrained=pretrained, with_head=True, with_backbone=True)

policy = tf.keras.mixed_precision.Policy("float32")
tf.keras.mixed_precision.set_global_policy(policy)
model.backbone.model = change_dtype(model.backbone.model, policy)

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
#print(measure(model, mode="onnx_gpu", input_shape=input_shape) - removal_gpu)
print(model.count_params())
