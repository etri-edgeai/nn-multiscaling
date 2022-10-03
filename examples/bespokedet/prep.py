import json

import tensorflow as tf
from tensorflow import keras

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection

def change_dtype_(model_dict, policy, distill_set=None):

    float32 = set()
    print(model_dict["config"]["output_layers"])
    for layer in model_dict["config"]["output_layers"]:
        print(layer)
        float32.add(layer[0])

    if distill_set is None:
        distill_set = set()

    for layer in model_dict["config"]["layers"]:
        if layer["class_name"] == "InputLayer":
            layer["config"]["dtype"] = "float16"
            continue
        if layer["config"]["name"] in float32 or layer["config"]["name"] in distill_set:
            continue
        elif layer["class_name"] == "Activation":
            continue
        elif layer["class_name"] == "Functional":
            change_dtype_(layer, policy, distill_set=distill_set)
        elif layer["class_name"] == "TFOpLambda" or layer["class_name"] == "AddLoss":
            continue
        else:
            layer["config"]["dtype"] = {'class_name': 'Policy', 'config': {'name': 'mixed_float16'}}


def change_dtype(model_, policy, distill_set=None, custom_objects=None):

    if type(model_) == keras.Sequential:
        input_layer = keras.layers.Input(batch_shape=model_.layers[0].input_shape, name="seq_input")
        prev_layer = input_layer
        for layer in model_.layers:
            layer._inbound_nodes = []
            prev_layer = layer(prev_layer)
        model_ = keras.models.Model([input_layer], [prev_layer])

    model_backup = model_
    model_dict = json.loads(model_.to_json())
    change_dtype_(model_dict, policy, distill_set=distill_set)
    model_json = json.dumps(model_dict)
    model_ = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
    for layer in model_backup.layers:
        model_.get_layer(layer.name).set_weights(layer.get_weights())
    return model_


def get_custom_objects():
    custom_objects = {
        "SimplePruningGate":SimplePruningGate,
        "StopGradientLayer":StopGradientLayer
    }
    return custom_objects
