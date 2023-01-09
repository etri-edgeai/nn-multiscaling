""" Profiler """

import os
import json
import copy
import shutil
import time
from os import listdir
from os.path import isfile, join
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()


from timeit import default_timer as timer
import yaml
import tensorflow as tf
import numpy as np
tf.random.set_seed(2)
import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
import sys
import argparse

tf.config.experimental.set_synchronous_execution(True)
tf.config.experimental.enable_op_determinism()

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for i, p in enumerate(physical_devices):
        tf.config.experimental.set_memory_growth(
            physical_devices[i], True
            )
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

from keras_flops import get_flops
import numpy as np
from tensorflow.keras.optimizers import Adam

from efficientnet.tfkeras import EfficientNetB0

import tf2onnx
import onnxruntime as rt

from bespoke.base.interface import ModelHouse
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
from nncompress.backend.tensorflow_.transformation.pruning_parser import StopGradientLayer
from nncompress.backend.tensorflow_.transformation.pruning_parser import has_intersection

from taskhandler import *

custom_objects = {
    "SimplePruningGate":SimplePruningGate,
    "StopGradientLayer":StopGradientLayer
}


BATCH_SIZE_GPU = 1
BATCH_SIZE_ONNX_GPU = 1
BATCH_SIZE_CPU = 1

def tf_convert_onnx(model, input_shape=None, output_path=None):
    """ tf -> onnx """
    if input_shape is None:
        input_shape = model.input.shape

    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
    if output_path is None:
        output_path = "/tmp/tmp_%d.onnx" % os.getpid()
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]
    return output_path, output_names

def tf_convert_tflite(model):
    """ tf -> tflite """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
    ]
    model_= converter.convert()
    return model_

def measure(model, mode="cpu", batch_size=-1, num_rounds=100, input_shape=None):
    """ Measure """
    total_t = 0
    if input_shape is None:
        if type(model.input) == list:
            input_shape = model.input[0].shape
        else:
            input_shape = list(model.input.shape)
        if input_shape[1] is None:
            input_shape = [None, 224, 224, 3]

    if batch_size == -1:
        if mode == "gpu":
            input_shape[0] = BATCH_SIZE_GPU
        elif mode == "onnx_gpu":
            input_shape[0] = BATCH_SIZE_ONNX_GPU
        else:
            input_shape[0] = BATCH_SIZE_CPU
    else:
        input_shape[0] = batch_size

    if mode == "cpu" and batch_size == -1:
        assert input_shape[0] == BATCH_SIZE_CPU

    tf.keras.backend.clear_session()
    input_shape = tuple(input_shape)
    if "onnx" in mode:
        output_path, output_names = tf_convert_onnx(model, input_shape=input_shape)
        if mode == "onnx_cpu":
            providers = ['CPUExecutionProvider']
            DEVICE_NAME = "cpu"
            DEVICE_INDEX = 0
        elif mode == "onnx_gpu":
            providers = [('CUDAExecutionProvider', {"device_id":0})]
            DEVICE_NAME = "cuda"
            DEVICE_INDEX = 0
        else:
            raise NotImplementedError("check your mode: %s" % mode)
        m = rt.InferenceSession(output_path, providers=providers)

        input_data = np.array(np.random.rand(*input_shape), dtype=np.float32)
        x_ortvalue = rt.OrtValue.ortvalue_from_numpy(input_data, DEVICE_NAME, DEVICE_INDEX)
        io_binding = m.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=x_ortvalue.device_name(),
            device_id=DEVICE_INDEX,
            element_type=input_data.dtype,
            shape=x_ortvalue.shape(),
            buffer_ptr=x_ortvalue.data_ptr())
        io_binding.bind_output(output_names[0])
        try:
            for i in range(10):
                #onnx_pred = m.run(output_names, {model.input.name: input_data})
                onnx_pred = m.run_with_iobinding(io_binding)

            for i in range(num_rounds):
                start = timer()
                #onnx_pred = m.run(output_names, {model.input.name: input_data})
                onnx_pred = m.run_with_iobinding(io_binding)
                time_ = float(timer() - start)
                total_t = total_t + time_
            avg_time = (total_t / float(num_rounds))
        except RuntimeError as re:
            avg_time =  1000000000.0

    elif mode == "gpu":
        # dummy run
        with tf.device("/gpu:0"):
            input_data = tf.convert_to_tensor(
                np.array(np.random.rand(*input_shape), dtype=np.float32), dtype=tf.float32)
            for i in range(10):
                model(input_data, training=False)

            for i in range(num_rounds):
                start = timer()
                model(input_data, training=False)
                time_ = float(timer() - start)
                total_t = total_t + time_
        avg_time = (total_t / float(num_rounds))

    elif mode == "cpu":
        with tf.device("/cpu:0"):
            input_data = tf.convert_to_tensor(
                np.array(np.random.rand(*input_shape), dtype=np.float32), dtype=tf.float32)
            for i in range(num_rounds):
                start = timer()
                model(input_data, training=False)
                time_ = float(timer() - start)
                total_t = total_t + time_
        avg_time = (total_t / float(num_rounds))
    elif mode == "tflite":
        tflite_model = tf_convert_tflite(model)
        # Save the model.
        with open('/tmp/tmp_%d.tflite' % os.getpid(), 'wb') as f:
          f.write(tflite_model)
        interpreter = tf.lite.Interpreter(model_path = "tmp.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        input_index = input_details[0]['index']

        total_t = 0
        input_data = np.array(np.random.rand(*input_shape), dtype=np.float32)
        for i in range(num_rounds):
            start = timer()
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            time_ = float(timer() - start)
            total_t = total_t + time_
        avg_time = (total_t / float(num_rounds))

    elif mode == "trt":
        from k2t import t2t_test
        avg_time = t2t_test(model, input_shape[0], num_rounds=num_rounds)
    else:

        raise NotImplementedError("!")

    del input_data
    return avg_time * 1000


def validate(config, model, detmodel):
    """ validation """
    backup = detmodel.backbone.model
    detmodel.backbone.model = model
    ret = validate_(config, detmodel)
    detmodel.backbone.model = backup
    return ret

def run():
    """ Run function """

    parser = argparse.ArgumentParser(description='Bespoke runner', add_help=False)
    parser.add_argument('--target_dir', type=str, default=None, help='model')
    parser.add_argument('--config', type=str, required=True) # dataset-sensitive configuration
    parser.add_argument('--notflite', action='store_true')
    parser.add_argument('--pretrained', type=str, default=None, help='model')
    parser.add_argument('--add', type=str, default=None, help='add [where]')
    parser.add_argument('--prefix', type=str, default=None, help='prefix')

    args = parser.parse_args()

    target_dir = args.target_dir
    node_file = os.path.join(target_dir, "nodes.json")
    mh = ModelHouse(None, custom_objects=custom_objects)
    mh.load(target_dir)

    if args.add is not None:
        if args.prefix is not None:
            prefix = args.prefix
        else:
            prefix = ""

        if not os.path.exists(node_file+"_backup_add"):
            shutil.copy(node_file, node_file+"_backup_add")

        with open(node_file, "r") as f:
            nodes = json.load(f)

            with open(args.add, "r") as f:
                to_add = json.load(f)

                for key, val in to_add["data"].items():
                    print(key)
                    node = nodes[key]

                    for metric in val:
                        node["profile"][prefix+metric] = val[metric]

            with open(node_file, "w") as f:
                json.dump(nodes, f, indent=4)

        return

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    for key in config:
        if hasattr(args, key):
            if config[key] in ["true", "false"]: # boolean handling.
                config[key] = config[key] == "true"
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
                if hvd.rank() == 0:
                    print("%s ---> %s" % (key, str(config[key])))

    config["task"] = build_config(config["task"])
    config["task"]["steps_per_execution"] = config["task"]["num_examples_per_epoch"] // config["task"]["batch_size"]
    config["task"]["steps_per_epoch"] = config["task"]["num_examples_per_epoch"] // config["task"]["batch_size"]

    notflite = args.notflite
    gpu_available = tf.test.is_gpu_available()

    detmodel = post_prep_(config["task"], mh._model, pretrained=args.pretrained, with_head=True)

    if not os.path.exists(node_file+"_backup"):
        shutil.copy(node_file, node_file+"_backup")

    with open(node_file, "r") as f:
        nodes = json.load(f)

    base = {
        "gpu": measure(mh.model, mode="gpu"),
        "cpu": measure(mh.model, mode="cpu"),
        #"onnx_gpu": measure(mh.model, mode="onnx_gpu"),
        "onnx_cpu": measure(mh.model, mode="onnx_cpu"),
    }
    data = {}
    base_acc = validate(config["task"], mh._model, detmodel)
    for key, val in nodes.items():

        print("-------", key, val["tag"])
        model_file = os.path.join(target_dir, "nets", key+".h5")
        nodes[key]["model_path"] = model_file

        node = mh.get_node(key)
        model = node.net.model
        
        profile = val["profile"]
        data[key] = profile

        if "app" in val["tag"]:
            is_gated = False
            for layer in model.layers:
                if layer.__class__ == SimplePruningGate:
                    is_gated = True
                    break
            
            if is_gated:
                mh.get_node(val["origin"]).net.wakeup()
                reference_model = mh.get_node(val["origin"]).net.model
                parser = PruningNNParser(
                    reference_model,
                    allow_input_pruning=True,
                    custom_objects=custom_objects,
                    gate_class=SimplePruningGate)
                parser.parse()
                model_ = parser.cut(model)
                print(model.count_params(), model_.count_params())
                model = model_
                print("ORIGIN:", data[val["origin"]])
                mh.get_node(val["origin"]).net.sleep()

        #if "flops" in profile:
        #    flops = profile["flops"]
        #else:
        flops = get_flops(model)
        profile["flops"] = flops

        if node.is_original():
            profile["iacc"] = float(base_acc)
        else:
            print("------------------------", key, "-----------------------------------")
            tf.keras.utils.plot_model(mh._parser._model, "backup.pdf")
            emodel = mh._parser.extract(mh.origin_nodes, [mh.get_node(key)])
            acc = validate(config["task"], emodel, detmodel)
            profile["iacc"] = float(acc)

        if not notflite:
            try:
                tflite = measure(model, mode="tflite")
            except RuntimeError as e:
                print(e)
                tflite = 100000000000.0

        gpu = 0
        cpu = 0
        onnx_gpu = measure(model, mode="onnx_gpu")
        onnx_cpu = measure(model, mode="onnx_cpu")

        profile["gpu"] = gpu
        profile["cpu"] = cpu
        profile["onnx_cpu"] = onnx_cpu
        profile["onnx_gpu"] = onnx_gpu

        if not notflite:
            profile["tflite"] = tflite
        print(key, profile, base)
        del model

    with open(node_file, "w") as f:
        json.dump(nodes, f, indent=4)

if __name__ == "__main__":
    run()
