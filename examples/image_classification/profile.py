import os
import json
import copy
import shutil
import time
from os import listdir
from os.path import isfile, join

from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
tf.random.set_seed(2)
import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
import sys

tf.config.experimental.set_synchronous_execution(True)
tf.config.experimental.enable_op_determinism()

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
    pass

from keras_flops import get_flops
import numpy as np
from tensorflow.keras.optimizers import Adam

from efficientnet.tfkeras import EfficientNetB0

import tf2onnx
import onnxruntime as rt

from bespoke.base.interface import ModelHouse
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection

BATCH_SIZE_GPU = 256
BATCH_SIZE_ONNX_GPU = 128
BATCH_SIZE_CPU = 1

def tf_convert_onnx(model):
    input_shape = model.input.shape
    spec = (tf.TensorSpec(input_shape, tf.float32, name=model.input.name),)
    output_path = "/tmp/tmp_%d.onnx" % os.getpid()
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]
    return output_path, output_names

def tf_convert_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
    ]
    model_= converter.convert()
    return model_

def measure(model, mode="cpu", batch_size=-1, num_rounds=100):
    total_t = 0
    input_shape = list(model.input.shape)
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
        output_path, output_names = tf_convert_onnx(model)
        if mode == "onnx_cpu":
            providers = ['CPUExecutionProvider']
            DEVICE_NAME = "cpu"
            DEVICE_INDEX = 0
        elif mode == "onnx_gpu":
            providers = [('CUDAExecutionProvider', {"device_id":1})]
            DEVICE_NAME = "cuda"
            DEVICE_INDEX = 1
        else:
            raise NotImplementedError("check your mode: %s" % mode)
        m = rt.InferenceSession(output_path, providers=providers)

        input_data = np.array(np.random.rand(*input_shape), dtype=np.float32)
        x_ortvalue = rt.OrtValue.ortvalue_from_numpy(input_data, DEVICE_NAME, DEVICE_INDEX)
        io_binding = m.io_binding()
        io_binding.bind_input(name=model.input.name, device_type=x_ortvalue.device_name(), device_id=DEVICE_INDEX, element_type=input_data.dtype, shape=x_ortvalue.shape(), buffer_ptr=x_ortvalue.data_ptr())
        io_binding.bind_output(output_names[0])
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

    elif mode == "gpu":
        # dummy run
        with tf.device("/gpu:0"):
            input_data = tf.convert_to_tensor(np.array(np.random.rand(*input_shape), dtype=np.float32), dtype=tf.float32)
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
            input_data = tf.convert_to_tensor(np.array(np.random.rand(*input_shape), dtype=np.float32), dtype=tf.float32)
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

def run():

    import sys
    hold_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        target_dir = sys.argv[2]
    else:
        target_dir = None

    ignore = False
    for arg in sys.argv:
        if "ignore" in arg:
            ignore = True
            break


    custom_objects = {
        "SimplePruningGate":SimplePruningGate,
        "StopGradientLayer":StopGradientLayer
    }

    noigpu = False
    for a in sys.argv:
        if a == "noigpu":
            print("NO IGPU TEST")
            noigpu = True
            break

    notflite = False
    for a in sys.argv:
        if a == "notflite":
            print("NO TFLITE TEST")
            notflite = True
            break

    gpu_available = tf.test.is_gpu_available()

    onlydir = [f for f in listdir(hold_dir) if not isfile(join(hold_dir, f))]
    for dir_ in onlydir:
        if target_dir != dir_ and target_dir is not None:
            continue
        dir_backup = dir_
        dir_ = hold_dir + "/" + dir_
        node_file = os.path.join(dir_, "nodes.json")
        print(dir_)

        mh = ModelHouse(None, custom_objects=custom_objects)
        mh.load(dir_)

        if not os.path.exists(node_file+"_backup"):
            shutil.copy(node_file, node_file+"_backup")

        with open(node_file, "r") as f:
            nodes = json.load(f)

        base = {
            "gpu": measure(mh.model, mode="gpu"),
            "cpu": measure(mh.model, mode="cpu"),
            "onnx_gpu": measure(mh.model, mode="onnx_gpu"),
            "onnx_cpu": measure(mh.model, mode="onnx_cpu"),
        }
        data = {}
        for key, val in nodes.items():
            model_file = os.path.join(dir_, "nets", key+".h5")
            nodes[key]["model_path"] = model_file

            node = mh.get_node(key)
            model = node.net.model
            
            profile = val["profile"]
            data[key] = profile

            if "app" in val["tag"]:
                #rmodel_path = os.path.join(dir_, "nets", val["origin"]+".h5")
                #reference_model = tf.keras.models.load_model(rmodel_path)
                reference_model = mh.get_node(val["origin"]).net.model
                parser = PruningNNParser(reference_model, allow_input_pruning=True, custom_objects=custom_objects, gate_class=SimplePruningGate)
                parser.parse()
                model_ = parser.cut(model)
                print(model.count_params(), model_.count_params())
                model = model_
                print("ORIGIN:", data[val["origin"]])

            #if "flops" in profile:
            #    flops = profile["flops"]
            #else:
            flops = get_flops(model)
            profile["flops"] = flops

            if flops > 390000000:
                gpu = 10000000000.0
                igpu = 10000000000.0
                tflite = 100000000000.0
            else:

                if not noigpu:
                    if node.is_original():
                        igpu = 0.0
                    else:
                        emodel = mh._parser.extract(mh.origin_nodes, [mh.get_node(key)])
                        igpu = measure(emodel, mode="gpu")
                        igpu = base["gpu"] - igpu
                gpu = measure(model, mode="gpu")
                if not notflite:
                    try:
                        tflite = measure(model, mode="tflite")
                    except Exception as e:
                        print(e)
                        tflite = 100000000000.0
            cpu = measure(model, mode="cpu")
            onnx_gpu = measure(model, mode="onnx_gpu")
            onnx_cpu = measure(model, mode="onnx_cpu")

            profile["gpu"] = gpu
            if not noigpu:
                profile["igpu"] = igpu
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
