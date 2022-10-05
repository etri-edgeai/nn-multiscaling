import os
import yaml
import argparse

from PIL import Image
import numpy as np

from automl.efficientdet.tf2 import infer_lib
from automl.efficientdet import hparams_config

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import StopGradientLayer
from taskhandler import *

def run():

    parser = argparse.ArgumentParser(description='Bespoke runner', add_help=False)
    parser.add_argument('--config', type=str, required=True) # dataset-sensitive configuration
    parser.add_argument('--pretrained', type=str, help='pretrained ckpt', required=True)
    parser.add_argument('--model_path', type=str, help='backend model path', required=True)
    parser.add_argument('--save_dir', type=str, help='save dir', required=True)
    parser.add_argument('--image', type=str, help='image path', required=True)

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    with open(args.config, 'r') as stream:
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

    model = tf.keras.models.load_model(args.model_path, custom_objects={"SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer})
    detmodel =  post_prep_infer_(config["task"], model, pretrained=args.pretrained, with_head=True)

    driver = infer_lib.ServingDriver.create(
        config["task"]["model_dir"], False, None, config["task"]["model_name"], 1, True, config_)
    driver.model = detmodel
    image_file = tf.io.read_file(args.image)
    image_arrays = tf.io.decode_image(
        image_file, channels=3, expand_animations=False)
    image_arrays = tf.expand_dims(image_arrays, axis=0)

    detections_bs = driver.serve(image_arrays)
    boxes, scores, classes, _ = tf.nest.map_structure(np.array, detections_bs)
    img = driver.visualize(
        np.array(image_arrays)[0],
        boxes[0],
        classes[0],
        scores[0],
        min_score_thresh=config_["nms_configs"]["score_thresh"] or 0.4,
        max_boxes_to_draw=config_["nms_configs"]["max_output_size"])
    output_image_path = os.path.join(args.save_dir, '0.jpg')
    Image.fromarray(img).save(output_image_path)
    print('writing file to %s' % output_image_path)

if __name__ == "__main__":
    run()
