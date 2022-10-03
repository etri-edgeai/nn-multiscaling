import os
import json
import shutil
import time
import argparse
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
    pass

first = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]

from profile import measure

parser = argparse.ArgumentParser(description='Bespoke runner', add_help=False)
parser.add_argument('--config', type=str, required=True) # dataset-sensitive configuration
parser.add_argument('--source_dir', type=str, help='model', required=True)
parser.add_argument('--postfix', type=str, help='model', default="good")
parser.add_argument('--base_value', type=float, help='model', required=True)
parser.add_argument('--obj_ratio', type=float,  help='model', required=True)
parser.add_argument('--lda', type=float, help='model', required=True)
parser.add_argument('--alter_ratio', type=float, help='model', required=True)
parser.add_argument('--metric', type=str, default="tflite", help='model', required=True)

args = parser.parse_args()

from efficientnet.tfkeras import EfficientNetB0
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection

def tf_convert_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
    ]
    model_= converter.convert()
    return model_

def compute_time(model_path, metric):
    model = tf.keras.models.load_model(model_path)
    print(model.count_params())
    if metric in ["gpu", "cpu", "tflite", "onnx_gpu", "onnx_cpu"]:
        return measure(model, mode=metric)
    else:
        from keras_flops import get_flops
        return get_flops(model)

dir_ = args.source_dir
model1 = dir_+"/students/gated_student%s.h5" % args.postfix
model2 = dir_+"/students/nongated_student%s.h5" % args.postfix

base_value = str(args.base_value)
obj_ratio = str(args.obj_ratio)
lda = str(args.lda)
metric = args.metric
alter_ratio = str(args.alter_ratio)
dataset_config = args.config

cmd = "PYTHONPATH='../..:./automl/efficientdet:$PYTHONPATH' CUDA_VISIBLE_DEVICES="+first+" python -u ../../run.py --config "+dataset_config+" --mode query_gated --source_dir "+ dir_ +" --sampling_ratio 1.0 --num_epochs 1 --step_ratio 0.3 --num_partitions 50 --num_imported_submodels 200 --num_approx 200 --postfix "+args.postfix+" --base_value "+base_value+" --obj_ratio "+obj_ratio+" --metric "+metric+" --lda "+lda+ " --alter_ratio "+alter_ratio
os.system(cmd)

print(compute_time(model2, metric))

print("finetune the following cmd!")
teacher_path = dir_ +"/base.h5"
cmd = "python -u ../../run.py --config "+dataset_config+" --mode finetune --model_path " + model1 + " --teacher_path " + teacher_path + " --sampling_ratio 1.0 --num_epochs 50 --step_ratio 0.3 --num_partitions 50 --num_imported_submodels 200 --num_approx 200 --postfix "+args.postfix+" --base_value 187.74 --obj_ratio 0.3 --metric tflite --lda 0.01"
print(cmd)
