import os
import json
import shutil
import time
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

dir_ = "experiments/imagenet_efnet2_200_200_02_sgd_approx/students"
model_file = os.path.join(dir_, "nongated_studenttest.h5")

from keras_flops import get_flops
model = tf.keras.models.load_model(model_file, custom_objects={"SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer})

#print(model.summary())
#from efficientnet.tfkeras import EfficientNetB2
#model = EfficientNetB2(weights='imagenet', input_shape=(260, 260, 3))

#from tensorflow.keras.applications import EfficientNetB0
#model = tf.keras.applications.efficientnet.EfficientNetB0(
#    include_top=True, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=1000,
#    classifier_activation='softmax')

print(get_flops(model))

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

#print(model.summary())

#print(measure(model, mode="trt"))
#print(measure(model, mode="gpu"))
#print(measure(model, mode="gpu", batch_size=1))
#print(measure(model, mode="cpu"))
#print(measure(model, mode="tflite", batch_size=1))
print(measure(model, mode="onnx_cpu") - removal_cpu)
#print(measure(model, mode="onnx_gpu") - removal_gpu)
print(model.count_params())
