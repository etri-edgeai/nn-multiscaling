"""
    Bespoke Runner
"""

from __future__ import print_function
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import json
import tempfile
import os
import gc
os.environ.pop('TF_CONFIG', None)
import copy
import traceback
import argparse
import shutil
import time
import subprocess
import pickle
import sys
if '.' not in sys.path:
  sys.path.insert(0, '.')

import horovod.tensorflow.keras as hvd
import horovod
from orderedset import OrderedSet
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB2
hvd.init()

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for i, p in enumerate(physical_devices):
        tf.config.experimental.set_memory_growth(
            physical_devices[i], True
            )
    tf.config.set_visible_devices(physical_devices[hvd.local_rank()], 'GPU')
tf.random.set_seed(2)
import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection

from bespoke.base.interface import ModelHouse
from bespoke.base.builder import RandomHouseBuilder
from bespoke import config as bespoke_config
from bespoke import backend as B

from taskhandler import *
import optimizer_factory
from get_imagenet_model import get_model


custom_objects = {
    "SimplePruningGate":SimplePruningGate,
    "StopGradientLayer":StopGradientLayer,
    "HvdMovingAverage":optimizer_factory.HvdMovingAverage
}

def student_model_save(model, dir_, inplace=False, prefix="", postfix=""):

    if inplace:
        student_house_path = dir_
    else:
        student_house_path = os.path.join(dir_, "students") 
        if not os.path.exists(student_house_path):
            os.mkdir(student_house_path)

    cnt = 0
    basename = prefix+"student"
    if postfix != "":
        filepath = os.path.join(student_house_path, "%s%s.h5" % (basename, postfix))
    else:
        filepath = os.path.join(student_house_path, "%s_%d.h5" % (basename, cnt))
        while os.path.exists(filepath):
            cnt += 1
            filepath = os.path.join(student_house_path, "%s_%d.h5" % (basename, cnt))
    tf.keras.models.save_model(model, filepath, overwrite=True, include_optimizer=False)
    print("model saving... %s done" % filepath)
    return filepath

def prep(config, model):
    # optimizer
    opt = Adam(lr=0.001)
    model.complie(opt, loss="categorical_crossentropy", run_eagerly=False)

def transfer_learning_(model_path, config_path, lr=0.1): 
    silence_tensorflow()
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(c) for c in list(range(1,num_gpus))])
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

    batch_size = config["batch_size"]
    model = tf.keras.models.load_model(model_path, custom_objects)
    model_ = B.make_transfer_model(model, output_idx, output_map, scale=1.0)

    train_(config["task"], "finetuned_"+basename, dirname, None, model_, is_tl=True)

    if hvd.size() > 1 and hvd.local_rank() == 0:
        student_model_save(model, dirname, inplace=True, prefix="finetuned_", postfix="ignore")


def transfer_learning(mh, config_path, target_dir=None, filter_=None, num_submodels_per_bunch=25):

    for n in mh.nodes:
        if not n.net.is_sleeping():
            n.sleep() # to_cpu 
    tf.keras.backend.clear_session()

    if filter_ is None:
        trainable_nodes = [n for n in mh.trainable_nodes]
    else:
        trainable_nodes = [n for n in mh.trainable_nodes if filter_(n)]

    # sort trainable_nodes by id
    trainable_nodes = sorted(trainable_nodes, key=lambda x: x.id_)

    to_remove = []
    while len(trainable_nodes) > 0:
        cnt = num_submodels_per_bunch 
        while True:
            if cnt == 0:
                to_remove.append(trainable_nodes.pop())
                cnt = num_submodels_per_bunch 

            tf.keras.backend.clear_session()
            if hvd.rank() == 0:
                print("Now training ... %d" % len(trainable_nodes))

            try:
                targets = []
                for i in range(len(trainable_nodes)):
                    if cnt == i:
                        break
                    targets.append(trainable_nodes[i])
                    targets[-1].wakeup()

            except Exception as e:
                print("Memory Problem Occurs %d" % cnt)
                gc.collect()
                for t in targets:
                    if not t.net.is_sleeping():
                        t.sleep()
                cnt -= 1
                continue

            try:
                house, output_idx, output_map = mh.make_train_model(targets, scale=1.0)

                #with tempfile.TemporaryDirectory() as dirpath:
                dirpath = "test"
                B.save_transfering_model(dirpath, house, output_idx, output_map)

                horovod.run(transfer_learning_, (os.path.join(dirpath, 'model.h5'), config_path), np=len(tf.config.list_physical_devices('GPU'))-1, use_mpi=True)

                if not os.path.exists(os.path.join(dirpath, f"finetuned_studentignore.h5")):
                    raise Exception("err")
                else:
                    # load and transfer model.
                    trmodel = tf.keras.models.load_model(os.path.join(dirpath, f"finetuned_studentignore.h5"), custom_objects=custom_objects)
                    for layer in house.layers:
                        if len(layer.get_weights()) > 0:
                            layer.set_weights(trmodel.get_layer(layer.name).get_weights())

                del house, output_idx, output_map
                gc.collect()

            except Exception as e:
                print(e)
                traceback.print_exc()
                print("Memory Problem Occurs %d" % cnt)
                if target_dir is not None:
                    tf.keras.utils.plot_model(house, os.path.join(target_dir, "hhhlarge.pdf"), expand_nested=True)
                print(tf.config.experimental.get_memory_info("GPU:0"))

                del house, output_idx, output_map
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
    parser.add_argument('--config', type=str, required=True) # dataset-sensitive configuration
    parser.add_argument('--mode', type=str, default="test", help='model')
    parser.add_argument('--source_dir', type=str, default=None, help='model')
    parser.add_argument('--target_dir', type=str, default=None, help='model')
    parser.add_argument('--model_path', type=str, default=None, help='model')
    parser.add_argument('--postfix', type=str, default="", help='model')
    parser.add_argument('--base_value', type=float, default=0, help='model')
    parser.add_argument('--obj_ratio', type=float, default=0.5, help='model')
    parser.add_argument('--lda', type=float, default=0.1, help='model')
    parser.add_argument('--alter_ratio', type=float, default=1.0, help='model')
    parser.add_argument('--metric', type=str, default="flops", help='model')
    parser.add_argument('--teacher_path', type=str, default=None, help='model')
    parser.add_argument('--trmode', action='store_true', help="for transfer learning")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--init', action='store_true')
    overriding_params = [
        ("sampling_ratio", float),
        ("lr", float),
        ("dloss_scale", float),
        ("memory_limit", float),
        ("params_limit", float),
        ("step_ratio", float),
        ("astep_ratio", float),
        ("num_partitions", int),
        ("num_imported_submodels", int),
        ("num_submodels_per_bunch", int),
        ("num_samples_for_profiling", int),
        ("num_epochs", int),
        ("num_approx", int),
    ]
    for name, type_ in overriding_params:
        parser.add_argument('--'+name, type=type_, default=None, help="method")
    args = parser.parse_args()

    if args.target_dir is None and args.mode in ["build", "approx", "transfer_learning", "profile"]:
        args.target_dir = args.source_dir + "_" + args.mode

    if hasattr(args, "target_dir") and args.target_dir is not None:
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
                if hvd.rank() == 0:
                    print("%s ---> %s" % (key, str(config[key])))

    if hasattr(args, "target_dir") and args.target_dir is not None:
        with open(os.path.join(args.target_dir, "args.log"), "w") as file_:
            json.dump(vars(args), file_)

        with open(os.path.join(args.target_dir, "config.log"), "w") as file_:
            json.dump(config, file_)

    config["task"] = build_config(config["task"])
    config["task"]["steps_per_execution"] = config["task"]["examples_per_epoch"] // config["task"]["batch_size"]
    config["task"]["steps_per_epoch"] = config["task"]["examples_per_epoch"] // config["task"]["batch_size"]

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


    #basemodel = extract_backbone(config["task"])
    if args.model_path is not None:
        if os.path.splitext(args.model_path)[1] != ".h5":
            model = get_model(args.model_path, config)
        else:
            model = tf.keras.models.load_model(args.model_path, custom_objects)

    assert model is not None

    # Make a model house
    if args.mode == "train":
        #model = post_prep_(config["task"], model)
        model = extract_backbone(config["task"])
        train_(config["task"], "test", args.target_dir, None, model)


    elif args.mode == "build":
        assert args.target_dir is not None
        assert model is not None
        mh = ModelHouse(model)
    elif args.mode != "test" and args.mode != "finetune" and args.mode != "cut":
        mh = ModelHouse(None, custom_objects=custom_objects)
        mh.load(args.source_dir)

    # Build if necessary
    filter_ = None
    if args.mode == "build":
        build_time_t1 = time.time()
        b = RandomHouseBuilder(mh)
        b.build(config["num_partitions"], config["step_ratio"])
        mh.build_base(
            model_list=config["models"],
            min_num=config["num_imported_submodels"],
            memory_limit=config["memory_limit"],
            params_limit=config["params_limit"],
            step_ratio=config["astep_ratio"])
        use_tl = True
        build_time_t2 = time.time()
        running_time["build_time"].append(build_time_t2 - build_time_t1)
        mh.save(args.target_dir)
    elif args.mode == "approx":
        approx_time_t1 = time.time()
        def f(n):
            return "app" in n.tag
        train_data_generator, _ = load_dataset_(config["task"])

        sample_inputs = []
        for x,y in train_data_generator:
            sample_inputs.append(x)
            if len(sample_inputs) > config["num_samples_for_profiling"]:
                break    
        mh.build_approx(
            min_num=config["num_approx"],
            memory_limit=config["memory_limit"],
            params_limit=config["params_limit"],
            init=args.init,
            data=sample_inputs,
            pruning_exit=config["pruning_exit"])
        mh.save(args.target_dir)
        use_tl = True
        filter_ = lambda n: "app" in n.tag
        approx_time_t2 = time.time()
        running_time["approx_time"].append(approx_time_t2 - approx_time_t1)
    else:
        use_tl = False

    batch_size = config["task"]["batch_size"]

    # Transfer learning
    if use_tl or args.mode == "transfer_learning":
        assert args.target_dir is not None
        tl_time_t1 = time.time()
        for n in mh.nodes:
            if args.init and args.mode == "build":
                if n.net.is_sleeping():
                    n.net.wakeup()
                if "alter" in n.tag:
                    gate_weights = {}
                    for layer in n.net.model.layers:
                        if type(layer) == SimplePruningGate:
                            gate_weights[layer.name] = layer.gates.numpy() 
                    n.net.model = tf.keras.models.clone_model(n.net.model)
                    for layer in n.net.model.layers:
                        if type(layer) == SimplePruningGate:
                            layer.gates.assign(gate_weights[layer.name])

            n.sleep() # to_cpu 

        transfer_learning(
            mh,
            args.config,
            target_dir=args.target_dir,
            num_submodels_per_bunch=config["num_submodels_per_bunch"],
            filter_=filter_,
        )
        tl_time_t2 = time.time()
        running_time["transfer_learning_time"].append(tl_time_t2 - tl_time_t1)
        if args.mode == "approx":
            profile_time_t1 = time.time()
            train_data_generator, _ = load_dataset_(config["task"])
            sample_inputs = []
            for x,y in train_data_generator:
                sample_inputs.append(x)
                if len(sample_inputs) > config["num_samples_for_profiling"]:
                    break
            mh.build_sample_data(sample_inputs)
            mh.profile()
            profile_time_t2 = time.time()
            running_time["profile_time"].append(profile_time_t2 - profile_time_t1)

        mh.save(args.target_dir)
        with open(os.path.join(args.target_dir, "running_time.log"), "w") as file_:
            json.dump(running_time, file_)

    elif args.mode == "test": # Test
        #model = post_prep_(config["task"], model)
        model = extract_backbone(config["task"])
        util_keras.restore_ckpt(
            model, "temp/emackpt-2", config["task"]["moving_average_decay"], skip_mismatch=False)
        validate_(config["task"], model)

    elif args.mode == "finetune": # Test

        finetune_time_t1 = time.time()
        # model must be a gated model
        dirname = os.path.dirname(args.model_path)
        basename = os.path.splitext(os.path.basename(args.model_path))[0] # ignore gated_ 
        if args.init:
            # backup gate
            gate_weights = {}
            for layer in model.layers:
                if type(layer) == SimplePruningGate:
                    gate_weights[layer.name] = layer.gates.numpy() 
            model = tf.keras.models.clone_model(model)
            for layer in model.layers:
                if type(layer) == SimplePruningGate:
                    layer.gates.assign(gate_weights[layer.name])

        distill_set = set()
        if args.teacher_path is not None:
            ex_map_filepath = os.path.join(dirname, basename+".map")
            if not os.path.exists(ex_map_filepath):
                ex_map_filepath = os.path.join(dirname, "non"+basename+".map")

            with open(ex_map_filepath, "r") as map_file:
                ex_maps = json.load(map_file)

            for locs in ex_maps:
                for loc in locs[1]:
                    distill_set.add(loc[0])
                    distill_set.add(loc[1])

        distil_loc = None
        if args.teacher_path is not None:
            teacher = tf.keras.models.load_model(args.teacher_path)

            ex_map_filepath = os.path.join(dirname, basename+".map")
            if not os.path.exists(ex_map_filepath):
                ex_map_filepath = os.path.join(dirname, "non"+basename+".map")

            with open(ex_map_filepath, "r") as map_file:
                ex_maps = json.load(map_file)

            distil_loc = []
            for locs in ex_maps:
                for loc in locs[1]:
                    distil_loc.append((loc[0], loc[1]))

            model_ = B.make_distiller(model, teacher, distil_loc, scale=config["dloss_scale"], model_builder=model_builder)
            distillation = True

        elif args.trmode:

            temp_path = os.path.dirname(args.model_path)
            with open(os.path.join(temp_path, "output_idx.pickle"), "rb") as f:
                output_idx = pickle.load(f)
            with open(os.path.join(temp_path, "output_map.pickle"), "rb") as f:
                output_map = pickle.load(f)

            model_ = B.make_transfer_model(model, output_idx, output_map, scale=1.0, model_builder=model_builder)
            distillation = True

        else:
            model_ = model
            distillation = False
            for layer in model.layers:
                layer.trainable = True

            if model_builder is not None:
                model_ = model_builder(model_.inputs, model_.outputs)

        train_(config["task"], "finetuned_"+basename, dirname, None, model_)

        if hvd.rank() == 0:
            filepath = student_model_save(model, dirname, inplace=True, prefix="finetuned_", postfix=args.postfix)
            finetune_time_t2 = time.time()
            running_time["finetune_time"].append(finetune_time_t2 - finetune_time_t1)
            running_time_dump(filepath, running_time)

    elif args.mode == "cut":
        if args.model_path is not None:
            basename = os.path.splitext(os.path.basename(args.model_path))[0] # ignore gated_ 
            dirname = os.path.dirname(args.model_path)
            reference_model = tf.keras.models.load_model(args.teacher_path)
        elif args.source_dir is not None:
            basename = "finetuned_student"+args.postfix
            model_path = os.path.join(args.source_dir, "students", basename+".h5")
            if not os.path.exists(model_path):
                basename = "gated_student"+args.postfix
                model_path = os.path.join(args.source_dir, "students", basename+".h5")
                if not os.path.exists(model_path):
                    raise ValueError("fintuned_gated or gated does not exist")
            model = tf.keras.models.load_model(model_path, custom_objects)
            dirname = os.path.join(args.source_dir, "students")
            reference_model = tf.keras.models.load_model(\
                os.path.join(args.source_dir, "students", "nongated_student"+args.postfix+".h5")
            )
        else:
            raise NotImplementedError("model_path or source_dir should be defined.")
        model_ = B.cut(model, reference_model, custom_objects)
        filepath = os.path.join(dirname, "cut_"+basename+args.postfix+".h5")
        filepath_plot = os.path.join(dirname, "cut_"+basename+args.postfix+".pdf")
        tf.keras.utils.plot_model(model_, filepath_plot, expand_nested=True, show_shapes=True)
        tf.keras.models.save_model(model_, filepath, overwrite=True)
        print(filepath, " is saved...")
        print(model_.count_params(), reference_model.count_params())

    elif args.mode == "profile": # Profiling
        assert args.target_dir is not None
        for n in mh.nodes:
            n.sleep() # to_cpu 
        train_data_generator, _ = load_dataset_(config["task"])
        sample_inputs = []
        for x,y in train_data_generator:
            sample_inputs.append(x)
            if len(sample_inputs) > config["num_samples_for_profiling"]:
                break
        mh.build_sample_data(sample_inputs)
        mh.profile()
        mh.save(args.target_dir)

    elif args.mode == "query":
        query_time_t1 = time.time()
        for n in mh.nodes:
            n.sleep() # to_cpu 
        ret = mh.select()
        filepath = student_model_save(ret, args.source_dir)
        query_time_t2 = time.time()
        running_time["query_time"].append(query_time_t2 - query_time_t1)
        running_time_dump(filepath, running_time)

    elif args.mode == "query_gated":
        query_time_t1 = time.time()
        for n in mh.nodes:
            n.sleep() # to_cpu 

        base_value = args.base_value
        assert base_value > 0
        obj_value = args.obj_ratio
        metric = args.metric
        gated, non_gated, ex_maps = mh.select((metric, base_value * obj_value, base_value), return_gated_model=True, lda=args.lda, ratio=args.alter_ratio)
        student_model_save(gated, args.source_dir, prefix="gated_", postfix=args.postfix, inplace=False)
        filepath = student_model_save(non_gated, args.source_dir, prefix="nongated_", postfix=args.postfix, inplace=False)
        student_dir = os.path.dirname(filepath)
        basename = os.path.splitext(os.path.basename(filepath))[0]
        with open(os.path.join(student_dir, "%s.map" % basename), "w") as file_:
            json.dump(ex_maps, file_)
        query_time_t2 = time.time()
        running_time["query_time"].append(query_time_t2 - query_time_t1)
        running_time_dump(filepath, running_time)
        
    else:
        raise NotImplementedError("Invalid mode %s" % args.mode)
 
if __name__ == "__main__":
    run()
