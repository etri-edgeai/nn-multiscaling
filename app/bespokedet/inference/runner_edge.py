# -*- coding: utf-8 -*-

from __future__ import print_function
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import time
import sys
import copy
import os

import numpy as np
import json
from PIL import Image
import numpy as np
import cv2
import base64
import time
import yaml
import scipy.misc

import tensorflow as tf
from onnx import ModelProto
import onnxruntime as rt

from automl.efficientdet import dataloader
from automl.efficientdet.tf2 import infer_lib, postprocess
from automl.efficientdet import hparams_config
from automl.efficientdet import coco_metric
from automl.efficientdet import utils

evaluator = None

def build_config(config):
    config_ = hparams_config.get_efficientdet_config(config["model_name"])
    config_.override(config, True)
    config_.image_size = utils.parse_image_size(config_.image_size)
    return config_.as_dict()

def preprocess(image_arrays, params, batch_size=1):

    def map_fn(image):
      input_processor = dataloader.DetectionInputProcessor(
          image, params['image_size'])
      input_processor.normalize_image(params['mean_rgb'],
                                      params['stddev_rgb'])
      input_processor.set_scale_factors_to_output_size()
      image = input_processor.resize_and_crop_image()
      image_scale = input_processor.image_scale_to_original
      return image, image_scale

    if batch_size:
      outputs = [map_fn(image_arrays[i]) for i in range(batch_size)]
      return [tf.stop_gradient(tf.stack(y)) for y in zip(*outputs)]

    return tf.vectorized_map(map_fn, image_arrays)

def postprocess_(params, outputs, scales):
    det_outputs = postprocess.postprocess_global(params, outputs[0],
                                                 outputs[1], scales)
    return det_outputs + tuple(outputs[2:])


def generate_detections_(params, outputs, labels):
    detections = postprocess.generate_detections(params, list(outputs[0]), list(outputs[1]), labels["image_scales"], labels["source_ids"])
    evaluator.update_state(labels["groundtruth_data"], postprocess.transform_detections(detections).numpy())


def get_engine():
    batch_size = 1
    config = "visdrone_edge.yaml"
    with open(config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    config_ = hparams_config.get_efficientdet_config(config["task"]["model_name"]).as_dict()

    config["task"] = build_config(config["task"])
    config["task"]["steps_per_execution"] = config["task"]["num_examples_per_epoch"] // config["task"]["batch_size"]
    config["task"]["steps_per_epoch"] = config["task"]["num_examples_per_epoch"] // config["task"]["batch_size"]

    for key in config_:
        if key in config["task"]:
            config_[key] = config["task"][key]

    driver = infer_lib.ServingDriver.create(
        config["task"]["model_dir"], False, None, config["task"]["model_name"], 1, True, config_)

    config_["val_file_pattern"] = config["task"]["val_file_pattern"]
    config_["use_fake_data"] = config["task"]["use_fake_data"]
    config_["max_instances_per_image"] = config["task"]["max_instances_per_image"]
    config_["debug"] = config["task"]["debug"]
    config_["batch_size"] = config["task"]["batch_size"]
    config_["eval_samples"] = config["task"]["eval_samples"]

    filepath = "compressed.onnx"
    providers = ['CPUExecutionProvider']
    DEVICE_NAME = "cpu"
    DEVICE_INDEX = 0

    with open(filepath, "rb") as f:
        model = ModelProto()
        model.ParseFromString(f.read())
        output_names = [n.name for n in model.graph.output]

        engine = (rt.InferenceSession(filepath, providers=providers), DEVICE_NAME, DEVICE_INDEX, output_names, driver)

    return engine, config_

def validate(val_json_path, tfrecord_path, dataset, engine, params):
    global evaluator
    evaluator = coco_metric.EvaluationMetric(
        filename=val_json_path, label_map=None)
    evaluator.reset_states()
    strategy = tf.distribute.get_strategy()
    count = params["eval_samples"]
    dataset_ = dataset.take(count)
    dataset_ = strategy.experimental_distribute_dataset(dataset_)
    for (images, labels) in dataset_:
        #strategy.run(_get_detections, (images, labels))
        outputs = predict(engine, params, images, True)
        generate_detections_(params, outputs, labels)       
    metrics = evaluator.result()

    eval_results = {}
    for i, name in enumerate(evaluator.metric_names):
        eval_results[name] = metrics[i]
    print(eval_results)


def visualize(engine, params):
    driver = engine[-1]

    image_file = tf.io.read_file("test_image.jpg")
    image_arrays = tf.io.decode_image(
        image_file, channels=3, expand_animations=False)
    image_arrays = tf.expand_dims(image_arrays, axis=0)
    #image_arrays = tf.cast(image_arrays, tf.float32)
    processed_image, scales = preprocess(image_arrays, params)

    boxes, scores, classes = predict(engine, params, processed_image, return_output=False, scales=scales)
    img = driver.visualize(
        np.array(image_arrays)[0],
        boxes[0],
        classes[0],
        scores[0],
        min_score_thresh=params["nms_configs"]["score_thresh"] or 0.4,
        max_boxes_to_draw=params["nms_configs"]["max_output_size"])
    output_image_path = 'test_output.jpg'
    print(img.shape)
    Image.fromarray(img).save(output_image_path)
    print('writing file to %s' % output_image_path)

def load_dataset(config):
    val_file_pattern = config["val_file_pattern"]
    use_fake_data = config["use_fake_data"]
    max_instances_per_image = config["max_instances_per_image"]
    debug = config["debug"]

    val_dataset = dataloader.InputReader(
        val_file_pattern,
        is_training=False,
        use_fake_data=use_fake_data,
        max_instances_per_image=max_instances_per_image,
        debug=debug)(
            copy.deepcopy(config))

    return val_dataset

def predict(engine, params, image, return_output=False, scales=None):
    m, DEVICE_NAME, DEVICE_INDEX, output_names, driver = engine
    im = image.numpy()
    inputs = {m.get_inputs()[0].name: im}
    out = m.run(None, inputs)
    out = [ out[:5], out[5:] ]

    if return_output:
        return out
    else:
        outputs = postprocess_(params, out, scales)
        boxes, scores, classes, _ = tf.nest.map_structure(np.array, outputs)
        return boxes, scores, classes


if __name__ == "__main__":
    engine, params = get_engine()
    val_json_path = "annotations_VisDrone_val.json"
    tfrecord_path = "tfrecord"
    dataset = load_dataset(params)
    validate(val_json_path, tfrecord_path, dataset, engine, params)
    visualize(engine, params)
