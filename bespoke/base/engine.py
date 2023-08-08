""" Bespoke Engine (Major functions)

"""

from __future__ import print_function
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import json
import tempfile
import os
import sys
import importlib.util

import copy
import time
import pickle
import traceback

import yaml
import numpy as np
import horovod
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
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB2

from bespoke import backend as B
from bespoke.base.task import TaskBuilder
from bespoke.base.interface import ModelHouse
from bespoke.base.builder import RandomHouseBuilder
from bespoke.train.train import train
from bespoke.utils.save import student_model_save

def module_load(task_path):
    """ Load python module in runtime

    """
    modulename = os.path.splitext(task_path)[0]
    spec = importlib.util.spec_from_file_location(modulename, task_path)
    task = importlib.util.module_from_spec(spec)
    sys.modules["modulename"] = task
    spec.loader.exec_module(task)

    for item in dir(task):
        if issubclass(getattr(task, item), TaskBuilder):
            return getattr(task, item)
    raise NotImplementedError("TaskBuilder class not found.")

def transfer_learning_(model_path, config_path, task_path): 
    """ Transfer learning function that is supposed to be executed via horovod

    """
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

    taskbuilder = module_load(task_path)(config)
    model = taskbuilder.load_model(model_path)
    model = taskbuilder.prep(model)
    data_gen = taskbuilder.load_dataset(is_tl=True)

    if config["use_amp"]:
        tf.keras.backend.set_floatx("float16")
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')

    model_ = B.make_transfer_model(model, output_idx, output_map, scale=1.0)
    train(
        config,
        model_,
        epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        data_gen=data_gen,
        is_tl=True,
        get_optimizer_gen=lambda config: taskbuilder.get_optimizer_gen(config),
        get_loss_gen=lambda config:taskbuilder.get_loss_gen(config),
        get_callbacks_gen=lambda config:taskbuilder.get_callbacks_gen(config),
        get_metrics_gen=lambda config:taskbuilder.get_metrics_gen(config),
        prefix="dummy",
        exclude_val=False,
        sampling_ratio=config["sampling_ratio"])

    if hvd.size() > 1 and hvd.local_rank() == 0:
        student_model_save(model, dirname, inplace=True, prefix="finetuned_", postfix="ignore")

def transfer_learning(
    mh, config_path, task_path, target_dir, taskbuilder, filter_=None, num_submodels_per_bunch=25, running_time=None):
    """Transfer learning

    """
    tl_time_t1 = time.time()

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

            except MemoryError as e:
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

            except RuntimeError as e:
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
                dirpath = "test"
                B.save_transfering_model(dirpath, house, output_idx, output_map)
                horovod.run(
                    transfer_learning_,
                    (os.path.join(dirpath, 'model.h5'), config_path, task_path),
                    np=len(tf.config.list_physical_devices('GPU'))-1,
                    use_mpi=True)

                if not os.path.exists(os.path.join(dirpath, f"finetuned_studentignore.h5")):
                    raise ValueError("err")
                else:
                    # load and transfer model.
                    trmodel = taskbuilder.load_model(os.path.join(dirpath, f"finetuned_studentignore.h5"))
                    for layer in house.layers:
                        if len(layer.get_weights()) > 0:
                            layer.set_weights(trmodel.get_layer(layer.name).get_weights())

                del house, output_idx, output_map
                import gc
                gc.collect()

            except ValueError as e:
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

    tl_time_t2 = time.time()
    if running_time is not None:
        running_time["transfer_learning_time"].append(tl_time_t2 - tl_time_t1)
        with open(os.path.join(target_dir, "running_time.log"), "w") as file_:
            json.dump(running_time, file_)

def build(model, target_dir, taskbuilder, running_time=None):
    """ Build function

    """
    config = taskbuilder.config
    mh = ModelHouse(model)
    build_time_t1 = time.time()
    b = RandomHouseBuilder(mh)
    b.build(config["num_partitions"], config["step_ratio"])
    mh.build_base(
        model_list=config["models"],
        min_num=config["num_imported_submodels"],
        memory_limit=config["memory_limit"],
        params_limit=config["params_limit"],
        step_ratio=config["astep_ratio"],
        use_last_types=config["use_last_types"])
    build_time_t2 = time.time()
    if running_time is not None:
        running_time["build_time"].append(build_time_t2 - build_time_t1)
        with open(os.path.join(target_dir, "running_time.log"), "w") as file_:
            json.dump(running_time, file_)
    mh.save(target_dir)
    return mh

def approximate(source_dir, target_dir, taskbuilder, running_time=None):
    """ Approximation function (Alternative Set Expansion)

    """
    config = taskbuilder.config
    mh = ModelHouse(None, custom_objects=taskbuilder.get_custom_objects())
    mh.load(source_dir)
    mh._model = taskbuilder.prep(mh._model)

    approx_time_t1 = time.time()
    def f(n):
        return "app" in n.tag
    train_data_generator = taskbuilder.load_dataset(split="train")

    sample_inputs = []
    for x,y in train_data_generator:
        sample_inputs.append(x)
        if len(sample_inputs) > config["num_samples_for_profiling"]:
            break 
    mh.build_approx(
        min_num=config["num_approx"],
        memory_limit=config["memory_limit"],
        params_limit=config["params_limit"],
        init=False,
        data=sample_inputs,
        pruning_exit=config["pruning_exit"])
    approx_time_t2 = time.time()
    if running_time is not None:
        running_time["approx_time"].append(approx_time_t2 - approx_time_t1)
    mh.save(target_dir)
    return mh

def finetune(model_path, taskbuilder, postfix="", teacher_path=None, running_time=None):
    """ Fintuneing

    """

    model = taskbuilder.load_model(model_path)

    finetune_time_t1 = time.time()
    # model must be a gated model
    dirname = os.path.dirname(model_path)
    basename = os.path.splitext(os.path.basename(model_path))[0] # ignore gated_ 

    distill_set = set()
    if teacher_path is not None:
        ex_map_filepath = os.path.join(dirname, basename+".map")
        if not os.path.exists(ex_map_filepath):
            ex_map_filepath = os.path.join(dirname, "non"+basename+".map")

        with open(ex_map_filepath, "r") as map_file:
            ex_maps = json.load(map_file)

        for locs in ex_maps:
            for loc in locs[1]:
                distill_set.add(loc[0])
                distill_set.add(loc[1])

    if taskbuilder.config["use_amp"]:
        tf.keras.backend.set_floatx("float16")
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')

    model = taskbuilder.prep(model)

    distil_loc = None
    if teacher_path is not None:
        teacher = tf.keras.models.load_model(teacher_path)
        teacher = taskbuilder.prep(teacher, is_teacher=True)

        ex_map_filepath = os.path.join(dirname, basename+".map")
        if not os.path.exists(ex_map_filepath):
            ex_map_filepath = os.path.join(dirname, "non"+basename+".map")

        with open(ex_map_filepath, "r") as map_file:
            ex_maps = json.load(map_file)

        distil_loc = []
        for locs in ex_maps:
            for loc in locs[1]:
                distil_loc.append((loc[0], loc[1]))

        model_ = B.make_distiller(model, teacher, distil_loc, scale=taskbuilder.config["dloss_scale"])

    else:
        model_ = model
        for layer in model.layers:
            layer.trainable = True

    data_gen = taskbuilder.load_dataset()

    train(
        taskbuilder.config,
        model_,
        epochs=taskbuilder.config["num_epochs"],
        batch_size=taskbuilder.config["batch_size"],
        data_gen=data_gen,
        is_distil=teacher_path is not None,
        get_optimizer_gen=lambda config: taskbuilder.get_optimizer_gen(config),
        get_loss_gen=lambda config:taskbuilder.get_loss_gen(config),
        get_callbacks_gen=lambda config:taskbuilder.get_callbacks_gen(config),
        get_metrics_gen=lambda config:taskbuilder.get_metrics_gen(config),
        prefix="finetuned",
        exclude_val=False,
        sampling_ratio=taskbuilder.config["sampling_ratio"],
        save_dir=dirname)

    if hvd.rank() == 0:
        filepath = student_model_save(model, dirname, inplace=True, prefix="finetuned_", postfix=postfix)
        finetune_time_t2 = time.time()
        if running_time is not None:
            running_time["finetune_time"].append(finetune_time_t2 - finetune_time_t1)
            with open(os.path.join(target_dir, "running_time.log"), "w") as file_:
                json.dump(running_time, file_)

def cut(model_path, teacher_path, taskbuilder, source_dir, postfix=""):
    """ Get a nongated model from a gated model

    """

    if model_path is not None:
        basename = os.path.splitext(os.path.basename(model_path))[0] # ignore gated_ 
        dirname = os.path.dirname(model_path)
        model = taskbuilder.load_model(model_path)
        reference_model = taskbuilder.load_model(teacher_path)
    elif source_dir is not None:
        basename = "finetuned_student"+postfix
        model_path = os.path.join(source_dir, "students", basename+".h5")
        if not os.path.exists(model_path):
            basename = "gated_student"+postfix
            model_path = os.path.join(source_dir, "students", basename+".h5")
            if not os.path.exists(model_path):
                raise ValueError("fintuned_gated or gated does not exist")
        model = taskbuilder.load_model(model_path)
        dirname = os.path.join(source_dir, "students")
        reference_model = taskbuilder.load_model(
            os.path.join(source_dir, "students", "nongated_student"+postfix+".h5"))
    else:
        raise NotImplementedError("model_path or source_dir should be defined.")
    model_ = B.cut(model, reference_model, taskbuilder.get_custom_objects())
    filepath = os.path.join(dirname, "cut_"+basename+postfix+".h5")
    filepath_plot = os.path.join(dirname, "cut_"+basename+postfix+".pdf")
    tf.keras.utils.plot_model(model_, filepath_plot, expand_nested=True, show_shapes=True)
    tf.keras.models.save_model(model_, filepath, overwrite=True)
    print(filepath, " is saved...")
    print(model_.count_params(), reference_model.count_params())

def query(mh, source_dir, base_value, obj_ratio=0.5, metric="cpu", lda=0.1, postfix="", running_time=None):
    """ Query function """

    query_time_t1 = time.time()
    for n in mh.nodes:
        n.sleep() # to_cpu 

    assert base_value > 0
    gated, non_gated, ex_maps = mh.select(
        (metric, base_value * obj_ratio, base_value), return_gated_model=True, lda=lda)
    student_model_save(gated, source_dir, prefix="gated_", postfix=postfix, inplace=False)
    filepath = student_model_save(non_gated, source_dir, prefix="nongated_", postfix=postfix, inplace=False)
    student_dir = os.path.dirname(filepath)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    with open(os.path.join(student_dir, "%s.map" % basename), "w") as file_:
        json.dump(ex_maps, file_)
    query_time_t2 = time.time()
    if running_time is not None:
        running_time["query_time"].append(query_time_t2 - query_time_t1)
        with open(os.path.join(source_dir, "running_time.log"), "w") as file_:
            json.dump(running_time, file_)
