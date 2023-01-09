from __future__ import print_function

import hashlib
import time
import os
import traceback
import json
import sys
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tf2onnx
import onnxruntime as rt

from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import preprocess_input

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection
from butils.optimizer_factory import HvdMovingAverage
from taskhandler import *
from prep import *

custom_objects = {
    "SimplePruningGate":SimplePruningGate,
    "StopGradientLayer":StopGradientLayer,
    "HvdMovingAverage":HvdMovingAverage
}

def remove_augmentation(model, custom_objects=None):
    found = False
    for l in model.layers:
        if l.name == "mixup_weight":
            found = True
            break
    if not found:
        return model

    model_dict = json.loads(model.to_json())

    to_removed_names = []
    to_removed = []
    image_name = None
    for layer in model_dict["config"]["layers"]:
        if layer["class_name"] == "InputLayer" or layer["config"]["name"] == "input_lambda":
            if "image" not in layer["config"]["name"]:
                to_removed_names.append(layer["config"]["name"])
                to_removed.append(layer)
            else:
                image_name = layer["config"]["name"]

    for layer in model_dict["config"]["layers"]:
        for inbound in layer["inbound_nodes"]:
            if type(inbound[0]) == str:
                if inbound[0] == "input_lambda":
                    inbound[0] = image_name
            else:
                for ib in inbound:
                    if ib[0] in to_removed_names: # input
                        ib[0] = image_name

    for r in to_removed:
        model_dict["config"]["layers"].remove(r)
    model_dict["config"]["input_layers"] = [[image_name, 0, 0]]

    model_json = json.dumps(model_dict)
    if custom_objects is None:
        custom_objects = {}
    model_ = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)

    for layer in model.layers:
        if layer.name in to_removed_names:
            continue
        model_.get_layer(layer.name).set_weights(layer.get_weights())

    return model_

def tf_convert_onnx(model, output_path):
    input_shape = model.input.shape
    spec = (tf.TensorSpec(input_shape, tf.float32, name=model.input.name),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]
    return output_path, output_names

def run():

    #path = "/home/jongryul/work/nn-multiscaling/examples/image_classification/experiments/imagenet_efnet2_200_200_02_sgd_newprune/nets"
    path = sys.argv[1]

    if os.path.isdir(path):

        onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        node_file = os.path.join(path, "..", "nodes.json")
        if os.path.exists(node_file):
            with open(node_file, "r") as f:
                nodes = json.load(f)

        onnxpath = "onnx_models"
        if not os.path.exists(onnxpath):
            os.mkdir(onnxpath)
        else:
            shutil.rmtree(onnxpath) 

        models = {}
        for filename in onlyfiles:
            if os.path.splitext(filename)[1] == ".h5":
                print(filename)
                filepath = os.path.join(path, filename)
                filename_base = os.path.splitext(filename)[0]
                model = tf.keras.models.load_model(filepath, custom_objects={"SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer})
                model = remove_augmentation(model, custom_objects=custom_objects)
                models[filename_base] = model

        meta = {}
        for filename in onlyfiles:
            if os.path.splitext(filename)[1] == ".h5":
                print(filename)
                filepath = os.path.join(path, filename)
                filename_base = os.path.splitext(filename)[0]

                model = models[filename_base]
                if os.path.exists(node_file):
                    val = nodes[filename_base]
                    if "app" in val["tag"]:
                        is_gated = False
                        for layer in model.layers:
                            if layer.__class__ == SimplePruningGate:
                                is_gated = True
                                break
                        
                        if is_gated:
                            reference_model = models[val["origin"]]
                            parser = PruningNNParser(reference_model, allow_input_pruning=True, custom_objects=custom_objects, gate_class=SimplePruningGate)
                            parser.parse()
                            model_ = parser.cut(model)
                            model = model_

                image_size = model.input.shape[1]
                onnx_filepath = os.path.join(onnxpath, filename_base+".onnx")
                tf_convert_onnx(model, onnx_filepath)

                meta[filename_base] = {
                    "basename": filename_base,
                    "onnxpath": onnx_filepath,
                    "tfpath": filepath,
                    "image_size": image_size
                }

        with open("meta.json", "w") as f:
            json.dump(meta, f)

    else:

        onnxpath = "./"
        
        filepath = path
        filename = os.path.basename(filepath)
        filename_base = os.path.splitext(filename)[0]
        model = tf.keras.models.load_model(filepath, custom_objects={"SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer})
        model = remove_augmentation(model, custom_objects=custom_objects)

        pretrained_path = sys.argv[2]
        config = sys.argv[3]

        with open(config, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

        config["task"] = build_config(config["task"])
        config["task"]["steps_per_execution"] = config["task"]["num_examples_per_epoch"] // config["task"]["batch_size"]
        config["task"]["steps_per_epoch"] = config["task"]["num_examples_per_epoch"] // config["task"]["batch_size"]

        model = post_prep_(config["task"], model, pretrained=pretrained_path, with_head=True, with_backbone=True)

        policy = tf.keras.mixed_precision.Policy("float32")
        tf.keras.mixed_precision.set_global_policy(policy)
        model.backbone.model = change_dtype(model.backbone.model, policy)

        onnx_filepath = os.path.join(onnxpath, filename_base+".onnx")
        tf_convert_onnx(model, onnx_filepath)

if __name__ == "__main__":

    run()
