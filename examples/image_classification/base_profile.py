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

from efficientnet.tfkeras import EfficientNetB0
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection

#dir_ = "experiments/efnet/cifar100_200_build_approx"
#model_file = os.path.join(dir_, "base.h5")

#model_file = "experiments/efnet/cub200_100_5_approx/students/cut_gated_studentgoodgood.h5"
#model_file = "experiments/efnet/cifar100_100_1_25_10_build_approx/students/cut_gated_studentgoodgood.h5"
model_file = "pretrained_weights/efnet2pretrain_cifar100_cifar100_model.046.h5"
#model_file = "pretrained_weights/efnetpretrained_teacher_cifar100_model.044.h5"
#model_file = "students/cifar100_100_1_build_approx_flops/cut_gated_studentgoodgood.h5"
#model_file = "/home/jongryul/work/nn-comp/examples/image_classification/saved_models/efnetv2b0efnetv2_cifar100_model.072.h5"
#model_file = "/home/jongryul/work/nn-comp/examples/image_classification/saved_models/efnetv2b0efnetv2_caltech_birds2011_model.044.h5"
#model_file = "/home/jongryul/work/nn-comp/examples/image_classification/saved_models/efnetv2b0efnetv2_oxford_iiit_pet_model.001.h5"

#model_file = "deepest/efnet_1_cifar100_True_curl_0.75_7300.h5"
#model_file = "deepest/efnet_1_caltech_birds2011_True_curl_0.75_8000.h5"
#model_file = "deepest/efnet_1_oxford_iiit_pet_True_curl_0.75_8000.h5"

#model_file = "pretrained_weights/efnetpretrained_teacher_cifar100_model.044.h5"
#model_file = "pretrained_weights/efnetpretrained_pet_oxford_iiit_pet_model.072.h5"

"""
model_file = "students/cifar100_100_5_approx_tflite/cut_gated_studentgoodgood.h5"
model_file = "students/cifar100_100_5_approx_flops/cut_gated_studentgoodgood.h5"
model_file = "students/cifar100_100_5_approx_flops/cut_gated_studentgoodgood.h5"
model_file = "students/pet_100_5_approx_tflite/cut_gated_studentgoodgood.h5"
"""

from keras_flops import get_flops
model = tf.keras.models.load_model(model_file)

print(get_flops(model))
#print(measure(model, mode="trt"))
print(measure(model, mode="gpu"))
#print(measure(model, mode="gpu", batch_size=1))
print(measure(model, mode="cpu"))
#print(measure(model, mode="tflite", batch_size=1))
print(measure(model, mode="onnx_cpu"))
print(measure(model, mode="onnx_gpu"))
