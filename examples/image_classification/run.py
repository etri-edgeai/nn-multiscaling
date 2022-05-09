from __future__ import print_function

import json
import os
import copy
import traceback
import argparse
import sys
import shutil

import tensorflow as tf
tf.random.set_seed(2)
import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from orderedset import OrderedSet

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection

from bespoke.base.interface import ModelHouse
from bespoke.base.builder import RandomHouseBuilder
from bespoke import config as bespoke_config

from models.loss import BespokeTaskLoss, accuracy
from train import train, iteration_based_train, load_data

MODELS = [
    tf.keras.applications.ResNet50V2,
    tf.keras.applications.InceptionResNetV2,
    tf.keras.applications.MobileNetV2,
    tf.keras.applications.MobileNet,
    tf.keras.applications.DenseNet121,
    tf.keras.applications.NASNetMobile,
    tf.keras.applications.EfficientNetB0,
    tf.keras.applications.EfficientNetB1,
    tf.keras.applications.EfficientNetB2,
    tf.keras.applications.EfficientNetV2B1,
    tf.keras.applications.EfficientNetV2S
]

def get_handler(model_name):
    if model_name == "efnetb0":
        from models import efficientnet as model_handler
    elif model_name == "resnet50":
        from models import resnet50 as model_handler
    else:
        raise ValueError("%s is not ready." % model_name)
    return model_handler

def student_model_save(model, dir_, finetune=False):

    student_house_path = os.path.join(dir_, "students") 
    if not os.path.exists(student_house_path):
        os.mkdir(student_house_path)

    cnt = 0
    if not finetune:
        basename = "student"
    else:
        basename = "fstudent"
    filepath = os.path.join(student_house_path, "%s_%d.h5" % (basename, cnt))
    while os.path.exists(filepath):
        cnt += 1
        filepath = os.path.join(student_house_path, "%s_%d.h5" % (basename, cnt))
    tf.keras.models.save_model(model, filepath, overwrite=True)
    print("model saving... done")


def validate(model, test_data_gen, model_handler):
    model_handler.compile(model, run_eagerly=True)
    return model.evaluate(test_data_gen, verbose=1)[1]

def finetune(dataset, model, model_handler, num_epochs, num_classes, sampling_ratio=1.0):
    for layer in model.layers:
        layer.trainable = True
    model_handler.compile(model, run_eagerly=True, transfer=False)
    train(dataset, model, "test", model_handler, num_epochs, callbacks=None, augment=True, n_classes=num_classes, sampling_ratio=sampling_ratio)
    

def transfer_learning(dataset, mh, model_handler, target_dir=None, filter_=None, num_submodels_per_bunch=25, num_epochs=3, num_classes=1000, sampling_ratio=0.1):
    loss = {mh.model.layers[-1].name:BespokeTaskLoss()}
    metrics={mh.model.layers[-1].name:accuracy}
    for n in mh.nodes:
        if not n.net.is_sleeping():
            n.sleep() # to_cpu 
    tf.keras.backend.clear_session()

    if filter_ is None:
        trainable_nodes = [n for n in mh.trainable_nodes]
    else:
        trainable_nodes = [n for n in mh.trainable_nodes if filter_(n)]
    to_remove = []
    while len(trainable_nodes) > 0:
        cnt = num_submodels_per_bunch 
        while True:
            if cnt == 0:
                to_remove.append(trainable_nodes.pop())
                cnt = num_submodels_per_bunch 

            tf.keras.backend.clear_session()
            print("Now training ... %d" % len(trainable_nodes))
            try:
                targets = []
                for i in range(len(trainable_nodes)):
                    if cnt == i:
                        break
                    targets.append(trainable_nodes[i])
                    targets[-1].wakeup()

            except Exception as e:
                print(e)
                print("Memory Problem Occurs %d" % cnt)
                print(tf.config.experimental.get_memory_info("GPU:0"))
                import gc
                gc.collect()
                for t in targets:
                    if not t.net.is_sleeping():
                        t.sleep()
                cnt -= 1
                continue

            try:
                house, output_idx, output_map = mh.make_train_model(targets, scale=1.0)
                model_handler.compile(house, run_eagerly=True, transfer=True, loss=loss, metrics=metrics)
                train(dataset, house, "test", model_handler, num_epochs, callbacks=None, augment=True, n_classes=num_classes, sampling_ratio=sampling_ratio)

                del house, output_idx, output_map
                import gc
                gc.collect()

            except Exception as e:
                print(e)
                traceback.print_exc()
                print("Memory Problem Occurs %d" % cnt)
                if target_dir is not None:
                    tf.keras.utils.plot_model(house, os.path.join(target_dir, "hhhlarge.pdf"), expand_nested=True)
                print(tf.config.experimental.get_memory_info("GPU:0"))

                del house, output_idx, output_map
                import gc
                gc.collect()
                for t in targets:
                    t.sleep()
                cnt -= 1
                continue

            # If the program runs here, the model has been traiend correctly.
            for t in targets:
                trainable_nodes.remove(t)
                t.sleep()
            break

    for n in to_remove:
        mh.remove(n)


def run():

    parser = argparse.ArgumentParser(description='Bespoke runner', add_help=False)
    parser.add_argument('--config', type=str) # dataset-sensitive configuration
    parser.add_argument('--mode', type=str, default="test", help='model')
    parser.add_argument('--source_dir', type=str, default=None, help='model')
    parser.add_argument('--target_dir', type=str, default=None, help='model')
    parser.add_argument('--model_name', type=str, default="model name for calling a handler", help='model')
    parser.add_argument('--model_path', type=str, default=None, help='model')
    parser.add_argument('--overwrite', action='store_true')
    overrding_params = [
        ("sampling_ratio", float),
        ("memory_limit", float),
        ("params_limit", float),
        ("step_ratio", float),
        ("batch_size_limit", int),
        ("num_partitions", int),
        ("num_imported_submodels", int),
        ("num_submodels_per_bunc", int),
        ("num_samples_for_profiling", int),
        ("num_epochs", int),
        ("num_approx", int)
    ]
    for name, type_ in overrding_params:
        parser.add_argument('--'+name, type=type_, default=None, help="method")
    args = parser.parse_args()

    model_handler = get_handler(args.model_name)

    if args.target_dir is not None:
        if os.path.exists(args.target_dir) and not args.overwrite:
            print("%s is already exists." % args.target_dir)
            sys.exit(1)
        else:
            if os.path.exists(args.target_dir):
                shutil.rmtree(args.target_dir)
            os.mkdir(args.target_dir)
    bespoke_config.TARGET_DIR = args.target_dir

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
                print("%s ---> %s" % (key, str(config[key])))

    if args.target_dir is not None:
        with open(os.path.join(args.target_dir, "args.log"), "w") as file_:
            json.dump(vars(args), file_)

        with open(os.path.join(args.target_dir, "config.log"), "w") as file_:
            json.dump(config, file_)

    if args.model_path is not None:
        if config["dataset"] == "imagenet2012" and args.mode != "finetune":
            model_class = None
            for model_ in MODELS:
                if args.model_path in model_.__name__:
                    model_class = model_
                    break
            model = model_class(include_top=True, weights="imagenet", classes=config["num_classes"])
        else:
            model = tf.keras.models.load_model(args.model_path)
    else:
        model = None

    # Make a model house
    if args.mode == "build":
        assert args.target_dir is not None
        assert model is not None
        mh = ModelHouse(model)
    elif args.mode != "test" and args.mode != "finetune":
        custom_objects = {
            "SimplePruningGate":SimplePruningGate,
            "StopGradientLayer":StopGradientLayer
        }
        mh = ModelHouse(None, custom_objects=custom_objects)
        mh.load(args.source_dir)
 
    # Build if necessary
    if args.mode == "build":
        b = RandomHouseBuilder(mh)
        b.build(config["num_partitions"])
        mh.build_base(
            model_list=config["models"],
            min_num=config["num_imported_submodels"],
            memory_limit=config["memory_limit"],
            params_limit=config["params_limit"])
        use_tl = True
    elif args.mode == "build_approx":
        def f(n):
            return "app" in n.tag
        mh.build_approx(
            min_num=config["num_approx"],
            memory_limit=config["memory_limit"],
            params_limit=config["params_limit"])
        use_tl = True
    else:
        use_tl = False

    # Transfer learning
    if use_tl or args.mode == "transfer_learning":
        for n in mh.nodes:
            n.sleep() # to_cpu 
        transfer_learning(
            config["dataset"],
            mh,
            model_handler,
            target_dir=args.target_dir,
            num_submodels_per_bunch=config["num_submodels_per_bunch"],
            num_epochs=config["num_epochs"],
            num_classes=config["num_classes"],
            sampling_ratio=config["sampling_ratio"]
        )
        mh.save(args.target_dir)

    elif args.mode == "test": # Test
        _, _, test_data_generator = load_data(
            config["dataset"],
            model_handler,
            sampling_ratio=1.0,
            training_augment=False,
            n_classes=config["num_classes"])
        ret = validate(model, test_data_gen, model_handler)
        print("Validation result: %f" % ret)

    elif args.mode == "finetune": # Test
        print(model.summary())
        finetune(config["dataset"], model, model_handler, config["num_epochs"], config["num_classes"], sampling_ratio=config["sampling_ratio"])
        dirname = os.path.dirname(args.model_path)
        student_model_save(model, dirname)

    elif args.mode == "profile": # Profiling
        for n in mh.nodes:
            n.sleep() # to_cpu 
        train_data_generator, _, _ = load_data(config["dataset"], model_handler, training_augment=True, n_classes=config["num_classes"])
        sample_inputs = []
        for x,y in train_data_generator:
            sample_inputs.append(x)
            if len(sample_inputs) > config["num_samples_for_profiling"]:
                break
        mh.build_sample_data(sample_inputs)
        mh.profile()
        mh.save(args.target_dir)

    elif args.mode == "query":
        for n in mh.nodes:
            n.sleep() # to_cpu 
        ret = mh.select()
        student_model_save(ret, args.source_dir)

    else:
        raise NotImplementedError("Invalid mode %s" % args.mode)
 
if __name__ == "__main__":
    run()
