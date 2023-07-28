from __future__ import print_function
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import json
import tempfile
import os

import sys

import copy
import time
import pickle

#os.environ['NCCL_P2P_LEVEL'] = "PIX"
import horovod.tensorflow.keras as hvd
hvd.init()

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for i, p in enumerate(physical_devices):
        tf.config.experimental.set_memory_growth(
            physical_devices[i], True
            )
    tf.config.set_visible_devices(physical_devices[hvd.local_rank()], 'GPU')

import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
import yaml


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection

custom_objects = {
    "SimplePruningGate":SimplePruningGate,
    "StopGradientLayer":StopGradientLayer,
    "HvdMovingAverage":optimizer_factory.HvdMovingAverage
}

from bespoke import backend as B

from train import train, load_dataset
from models.custom import GAModel
from prep import add_augmentation, change_dtype
from utils import optimizer_factory

from efficientnet.tfkeras import EfficientNetB0, EfficientNetB2


def transfer_learning_(model_path, model_name, config_path, lr=0.1 augmentify=None): 
    silence_tensorflow()
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(c) for c in list(range(1,num_gpus+1))])
    hvd.init()
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for i, p in enumerate(physical_devices):
            tf.config.experimental.set_memory_growth(
                physical_devices[i], True
                )
        tf.config.set_visible_devices(physical_devices[hvd.local_rank()+1], 'GPU')
    tf.random.set_seed(2)
    random.seed(1234)
    np.random.seed(1234)
    model_handler = get_handler(model_name)

    dirname = os.path.dirname(model_path)
    basename = os.path.splitext(os.path.basename(model_path))[0] # ignore gated_ 

    temp_path = os.path.dirname(model_path)
    with open(os.path.join(temp_path, "output_idx.pickle"), "rb") as f:
        output_idx = pickle.load(f)
    with open(os.path.join(temp_path, "output_map.pickle"), "rb") as f:
        output_map = pickle.load(f)

    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    batch_size = model_handler.get_batch_size(config["dataset"])
    model = tf.keras.models.load_model(model_path, custom_objects)

    if config["training_conf"]["use_amp"]:
        tf.keras.backend.set_floatx("float16")
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        model = change_dtype(model, mixed_precision.global_policy(), custom_objects=custom_objects)

    #model = add_augmentation(model, model_handler.width, train_batch_size=batch_size, do_mixup=config["training_conf"]["mixup_alpha"] > 0, do_cutmix=config["training_conf"]["cutmix_alpha"] > 0, custom_objects=custom_objects, update_batch_size=True)

    if "training_conf" in config and config["training_conf"]["grad_accum_steps"] > 1:
        model_builder = lambda x, y: GAModel(
            config["training_conf"]["use_amp"],
            config["training_conf"]["hvd_fp16_compression"],
            config["training_conf"]["grad_clip_norm"],
            config["training_conf"]["grad_accum_steps"],
            x, y
            )
    else:
        model_builder = None

    model_ = B.make_transfer_model(model, output_idx, output_map, scale=1.0, model_builder=model_builder)

    if "training_conf" in config:
        tconf = config["training_conf"]
        tconf["mode"] = "distillation"
    else:
        tconf = None

    train(
        config["dataset"],
        model_,
        "finetuned_"+basename,
        model_handler,
        config["num_epochs"],
        augment=True,
        n_classes=config["num_classes"],
        sampling_ratio=config["sampling_ratio"],
        save_dir=dirname,
        conf=tconf)

    if hvd.size() > 1 and hvd.local_rank() == 0:
        student_model_save(model, dirname, inplace=True, prefix="finetuned_", postfix="ignore")
