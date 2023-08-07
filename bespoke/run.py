from __future__ import print_function
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import json
import tempfile
import os
os.environ.pop('TF_CONFIG', None)

import sys
if '.' not in sys.path:
  sys.path.insert(0, '.')

import copy
import traceback
import argparse
import shutil
import time
import subprocess
import pickle
import yaml

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from bespoke.base.interface import ModelHouse
from bespoke import config as bespoke_config
from bespoke import backend as B
from bespoke.base.engine import module_load, transfer_learning, build, approximate, finetune, cut, query

def run():

    parser = argparse.ArgumentParser(description='Bespoke runner', add_help=False)
    parser.add_argument('--config', type=str) # dataset-sensitive configuration
    parser.add_argument('--mode', type=str, default="test", help='model')
    parser.add_argument('--source_dir', type=str, default=None, help='model')
    parser.add_argument('--target_dir', type=str, default=None, help='model')
    parser.add_argument('--model_name', type=str, default="model name for calling a handler", help='model')
    parser.add_argument('--model_path', type=str, default=None, help='model')
    parser.add_argument('--postfix', type=str, default="", help='model')
    parser.add_argument('--base_value', type=float, default=0, help='model')
    parser.add_argument('--obj_ratio', type=float, default=0.5, help='model')
    parser.add_argument('--lda', type=float, default=0.1, help='model')
    parser.add_argument('--alter_ratio', type=float, default=1.0, help='model')
    parser.add_argument('--metric', type=str, default="flops", help='model')
    parser.add_argument('--teacher_path', type=str, default=None, help='model')
    parser.add_argument('--overwrite', action='store_true')
    overriding_params = [
        ("sampling_ratio", float),
        ("lr", float),
        ("dloss_scale", float),
        ("memory_limit", float),
        ("params_limit", float),
        ("step_ratio", float),
        ("astep_ratio", float),
        ("batch_size_limit", int),
        ("num_partitions", int),
        ("num_imported_submodels", int),
        ("num_submodels_per_bunch", int),
        ("num_samples_for_profiling", int),
        ("num_epochs", int),
        ("num_approx", int),
        ("use_last_types", bool),
    ]
    for name, type_ in overriding_params:
        if type_ != bool:
            parser.add_argument('--'+name, type=type_, default=None, help="method")
        else:
            parser.add_argument('--'+name, action='store_true', help="Binary decision")
    args = parser.parse_args()

    if args.target_dir is None and args.mode in ["build", "approx", "transfer_learning"]:
        args.target_dir = args.source_dir + "_" + args.mode

    if hasattr(args, "target_dir") and args.target_dir is not None:
        if os.path.exists(args.target_dir) and not args.overwrite:
            print("%s is already exists." % args.target_dir)
            sys.exit(1)
        else:
            if os.path.exists(args.target_dir):
                shutil.rmtree(args.target_dir)
            os.mkdir(args.target_dir)

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
    # debug
    for key, type_ in overriding_params:
        assert key in config

    task_class = module_load(config["taskbuilder"])
    taskbuilder = task_class(config)

    if hasattr(args, "target_dir") and args.target_dir is not None:
        with open(os.path.join(args.target_dir, "args.log"), "w") as file_:
            json.dump(vars(args), file_)

        with open(os.path.join(args.target_dir, "config.log"), "w") as file_:
            json.dump(config, file_)

    if args.source_dir is not None:
        if os.path.exists(os.path.join(args.source_dir, "running_time.log")):
            with open(os.path.join(args.source_dir, "running_time.log"), "r") as file_:
                running_time = json.load(file_)
        else:
            running_time = {
                "transfer_learning_time":[],
                "build_time":[],
                "approx_time":[],
                "query_time":[],
                "finetune_time":[],
                "profile_time":[]
            }
    else:
        running_time = {
            "transfer_learning_time":[],
            "build_time":[],
            "approx_time":[],
            "query_time":[],
            "finetune_time":[],
            "profile_time":[]
        }

    if args.model_path is not None:
        model = taskbuilder.load_model(args.model_path)
        assert model is not None
    else:
        model = None

    # Make a model house
    if args.mode == "build":
        assert args.target_dir is not None
        assert model is not None
        model = taskbuilder.prep(model)
    elif args.mode in ["approximate", "query", "transfer_learning", "profile"]:
        mh = ModelHouse(None, custom_objects=taskbuilder.get_custom_objects())
        mh.load(args.source_dir)
        mh._model = taskbuilder.prep(mh._model)

    # Build if necessary
    filter_ = None
    if args.mode == "build":
        mh = build(model, args.target_dir, taskbuilder, running_time=running_time)

        transfer_learning(
            mh,
            args.config,
            config["taskbuilder"],
            target_dir=args.target_dir,
            taskbuilder=taskbuilder,
            num_submodels_per_bunch=config["num_submodels_per_bunch"],
            filter_=filter_,
            running_time=running_time
        )
        mh.save(args.target_dir)

    elif args.mode == "approx":
        filter_ = lambda n: "app" in n.tag
        mh = approximate(args.source_dir, args.target_dir, taskbuilder, running_time=running_time) 

        transfer_learning(
            mh,
            args.config,
            config["taskbuilder"],
            target_dir=args.target_dir,
            taskbuilder=taskbuilder,
            num_submodels_per_bunch=config["num_submodels_per_bunch"],
            filter_=filter_,
            running_time=running_time
        )
        mh.save(args.target_dir)

    elif args.mode == "profile":
        for n in mh.nodes:
            n.sleep() # to_cpu 
        train_data_generator = taskbuilder.load_dataset(split="train")
        sample_inputs = []
        for x,y in train_data_generator:
            sample_inputs.append(x)
            if len(sample_inputs) > config["num_samples_for_profiling"]:
                break
        mh.build_sample_data(sample_inputs)
        mh.profile()
        mh.save(args.target_dir)

    elif args.mode == "transfer_learning":
        assert args.target_dir is not None
        transfer_learning(
            mh,
            args.config,
            config["taskbuilder"],
            target_dir=args.target_dir,
            taskbuilder=taskbuilder,
            num_submodels_per_bunch=config["num_submodels_per_bunch"],
            filter_=filter_,
            running_time=running_time
        )
        mh.save(args.target_dir)

    elif args.mode == "test": # Test
        model = taskbuilder.prep(model)
        test_data_gen = taskbuilder.load_dataset(split="test")
        taskbuilder.compile(model)
        ret = model.evaluate(test_data_gen, verbose=1)[1]
        print("Validation result: %f" % ret)

    elif args.mode == "finetune": # Test
        finetune(
            args.model_path,
            taskbuilder,
            postfix=args.postfix,
            teacher_path=args.teacher_path,
            running_time=running_time)

    elif args.mode == "cut":
        cut(args.model_path, args.teacher_path, taskbuilder, args.source_dir, postfix=args.postfix)

    elif args.mode == "query":
        query(
            mh,
            args.source_dir,
            args.base_value,
            obj_ratio=args.obj_ratio,
            metric=args.metric,
            lda=args.lda,
            postfix=args.postfix, running_time=running_time)
        
    else:
        raise NotImplementedError("Invalid mode %s" % args.mode)
 
if __name__ == "__main__":
    run()
