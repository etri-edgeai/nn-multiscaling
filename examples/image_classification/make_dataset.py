import os
import json
import copy
import shutil
import time
import yaml
from os import listdir
from os.path import isfile, join
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tempfile

from timeit import default_timer as timer

import tensorflow as tf
import horovod
import numpy as np
tf.random.set_seed(2)
import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
import sys
import argparse

import horovod.tensorflow.keras as hvd
hvd.init()

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for i, p in enumerate(physical_devices):
        tf.config.experimental.set_memory_growth(
            physical_devices[i], True
            )
    tf.config.set_visible_devices(physical_devices[hvd.local_rank()], 'GPU')


from keras_flops import get_flops
import numpy as np
from tensorflow.keras.optimizers import Adam

from efficientnet.tfkeras import EfficientNetB0

from bespoke.base.interface import ModelHouse
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection
from prep import remove_augmentation, add_augmentation, change_dtype

from train import load_dataset, train
from run import get_handler
from models.custom import GAModel
from bespoke import backend as B


custom_objects = {
    "SimplePruningGate":SimplePruningGate,
    "StopGradientLayer":StopGradientLayer
}


def validate(model, model_handler, dataset):
    custom_objects = {
        "SimplePruningGate":SimplePruningGate,
        "StopGradientLayer":StopGradientLayer
    }
    if dataset == "imagenet2012":
        n_classes = 1000
    elif dataset == "cifar100":
        n_classes = 100

    batch_size = model_handler.batch_size
    model = add_augmentation(model, model_handler.width, train_batch_size=batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_objects)
    (_, _, test_data_generator), (iters, iters_val) = load_dataset(
        dataset,
        model_handler,
        training_augment=False,
        n_classes=n_classes)
    model_handler.compile(model, run_eagerly=False)
    return model.evaluate(test_data_generator, verbose=1)[1]

def finetune(model_path, teacher_path, targets, model_name, config_path, epochs, lr=0.1):
    silence_tensorflow()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    hvd.init()
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for i, p in enumerate(physical_devices):
            tf.config.experimental.set_memory_growth(
                physical_devices[i], True
                )
        tf.config.set_visible_devices(physical_devices[hvd.local_rank()], 'GPU')
    tf.random.set_seed(2)
    random.seed(1234)
    np.random.seed(1234)
    model_handler = get_handler(model_name)

    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

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

    batch_size = model_handler.get_batch_size(config["dataset"])

    distill_set = set()
    dirname = os.path.dirname(model_path)
    basename = os.path.splitext(os.path.basename(model_path))[0] # ignore gated_ 
    ex_map_filepath = os.path.join(dirname, basename+".map")
    with open(ex_map_filepath, "r") as map_file:
        ex_maps = json.load(map_file)

    for locs in ex_maps:
        for loc in locs[1]:
            distill_set.add(loc[0])
            distill_set.add(loc[1])

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    if config["training_conf"]["use_amp"]:
        tf.keras.backend.set_floatx("float16")
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        model = change_dtype(model, mixed_precision.global_policy(), custom_objects=custom_objects, distill_set=distill_set)

    model = add_augmentation(model, model_handler.width, train_batch_size=batch_size, do_mixup=True, do_cutmix=True, custom_objects=custom_objects)

    teacher = tf.keras.models.load_model(teacher_path, custom_objects=custom_objects)
    teacher = remove_augmentation(teacher, custom_objects=custom_objects)
    if config["training_conf"]["use_amp"]:
        teacher = change_dtype(teacher, mixed_precision.global_policy(), custom_objects=custom_objects, distill_set=distill_set)

    ex_map_filepath = os.path.join(dirname, basename+".map")
    with open(ex_map_filepath, "r") as map_file:
        ex_maps = json.load(map_file)

    distil_loc = []
    for locs in ex_maps:
        for loc in locs[1]:
            distil_loc.append((loc[0], loc[1]))

    model_ = B.make_distiller(model, teacher, distil_loc, scale=config["dloss_scale"], model_builder=model_builder)

    if "training_conf" in config:
        tconf = config["training_conf"]
    else:
        tconf = None

    tconf["mode"] = "distillation"
    train(
        config["dataset"],
        model_,
        "finetuned_"+basename,
        model_handler,
        epochs,
        augment=True,
        n_classes=config["num_classes"],
        sampling_ratio=config["sampling_ratio"],
        save_dir=dirname,
        conf=tconf)

    if hvd.size() > 1 and hvd.local_rank() == 0:
        filepath = os.path.join(dirname, "ret.h5" )
        tf.keras.models.save_model(model, filepath, overwrite=True, include_optimizer=False)

def run():

    parser = argparse.ArgumentParser(description='Bespoke runner', add_help=False)
    parser.add_argument('--config', type=str) # dataset-sensitive configuration
    parser.add_argument('--target_dir', type=str, default=None, help='model')
    parser.add_argument('--model_name', type=str, default="model name for calling a handler", help='model')
    parser.add_argument('--epochs', type=int, default=1, help="number of data")
    parser.add_argument('--num', type=int, default=100, help="number of data")

    args = parser.parse_args()
    from run import get_handler
    model_handler = get_handler(args.model_name)

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    target_dir = args.target_dir
    node_file = os.path.join(target_dir, "nodes.json")

    mh = ModelHouse(None, custom_objects=custom_objects)
    mh.load(target_dir)

    if not os.path.exists(node_file+"_backup"):
        shutil.copy(node_file, node_file+"_backup")

    with open(node_file, "r") as f:
        nodes = json.load(f)

    anodes = [
        key
        for key, value in nodes.items() if not mh.get_node(key).is_original()
    ]


    records = []
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    with tempfile.TemporaryDirectory() as dirpath:
        tf.keras.models.save_model(mh._model, os.path.join(dirpath, "base.h5"), overwrite=True, include_optimizer=False)
        for i in range(args.num):
            print("-------------------------- %d ----------------------" % i)
            r = np.random.randint(len(anodes)+1)
            targets = np.random.choice(anodes, r, replace=False)
            np.random.shuffle(targets) # in-place

            targets_ = []
            for n in targets:
                compat = True
                for m in targets_:
                    if not mh._parser.is_compatible(mh.get_node(n), mh.get_node(m)):
                        compat = False
                        break
                if compat:
                    targets_.append(n)

            target_nodes = [
                mh.get_node(t)
                for t in targets_
            ]
            gated, non_gated, ex_maps = mh._parser.extract(mh.origin_nodes, target_nodes, return_gated_model=True)

            filepath = os.path.join(dirpath, "current.h5")
            tf.keras.models.save_model(gated, filepath, overwrite=True, include_optimizer=False)
            with open(os.path.join(dirpath, "current.map"), "w") as file_:
                json.dump(ex_maps, file_)

            acc = validate(gated, model_handler, config["dataset"])
            print(acc)
            horovod.run(finetune, (filepath, os.path.join(dirpath, "base.h5"), targets_, args.model_name, args.config, args.epochs, 0.1), np=3, use_mpi=True)

            ret = tf.keras.models.load_model(os.path.join(dirpath, "ret.h5"), custom_objects=custom_objects)
            acc = validate(ret, model_handler, config["dataset"])
            print(targets_, acc)
            records.append(
                (targets_, acc)
            )

            with open("dataset/pp.json", "w") as f:
                json.dump(records, f)

if __name__ == "__main__":
    run()
