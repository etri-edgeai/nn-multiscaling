import math
import os
import logging

from tqdm import tqdm
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import horovod.tensorflow.keras as hvd

from datagen_ds import DataGenerator

from dataloader import dataset_factory
from utils import callbacks as custom_callbacks
from utils import optimizer_factory

from models.loss import BespokeTaskLoss, accuracy

def load_data(dataset, model_handler, sampling_ratio=1.0, training_augment=True, batch_size=-1, n_classes=100):

    dim = (model_handler.height, model_handler.width)
    preprocess_func = model_handler.preprocess_func
    if hasattr(model_handler, "batch_preprocess_func"):
        batch_pf = model_handler.batch_preprocess_func
    else:
        batch_pf = None

    if hasattr(model_handler, "parse_fn"):
        parse_fn = model_handler.parse_fn
    else:
        parse_fn = None

    if batch_size == -1:
        batch_size_ = model_handler.get_batch_size(dataset)
    else:
        batch_size_ = batch_size

    augment = True
    reg_augment = True

    if dataset == "imagenet2012":
        ds_train = tfds.load(dataset, split="train")
        ds_val = tfds.load(dataset, split="validation")
    else:
        ds_train = tfds.load(dataset, split="train")
        ds_val = tfds.load(dataset, split="test")
    train_examples = None
    val_examples = None
    is_batched = False

    train_data_generator = DataGenerator(
        ds_train,
        dataset=dataset,
        batch_size=batch_size_,
        augment=training_augment and augment,
        reg_augment=training_augment and reg_augment,
        dim=dim,
        n_classes=n_classes,
        n_examples=train_examples,
        preprocess_func=preprocess_func,
        is_batched=is_batched,
        batch_preprocess_func=batch_pf,
        parse_fn=parse_fn,
        sampling_ratio=sampling_ratio)

    valid_data_generator = DataGenerator(
        ds_val,
        dataset=dataset,
        batch_size=batch_size_,
        augment=False,
        dim=dim,
        shuffle=False,
        n_classes=n_classes,
        n_examples=val_examples,
        preprocess_func=preprocess_func,
        is_batched=is_batched,
        batch_preprocess_func=batch_pf,
        parse_fn=parse_fn)

    test_data_generator = DataGenerator(
        ds_val,
        dataset=dataset,
        batch_size=batch_size_,
        augment=False,
        dim=dim,
        shuffle=False,
        n_classes=n_classes,
        n_examples=val_examples,
        preprocess_func=preprocess_func,
        is_batched=is_batched,
        batch_preprocess_func=batch_pf,
        parse_fn=parse_fn)

    return train_data_generator, valid_data_generator, test_data_generator


def load_data_nvidia(dataset, model_handler, sampling_ratio=1.0, training_augment=True, batch_size=-1, n_classes=100, cutmix_alpha=1.0, mixup_alpha=0.8):

    if dataset == "imagenet2012":
        data_dir = "tensorflow_datasets/imagenet2012/5.1.0_dali"
    elif dataset == "cifar100":
        data_dir = "tensorflow_datasets/cifar100/3.0.2_dali"
    else:
        raise NotImplementedError("no support for the other datasets")

    dim = (model_handler.height, model_handler.width)

    if batch_size == -1:
        batch_size = model_handler.get_batch_size(dataset)

    augmenter = "autoaugment"
    augmenter = None
    augmenter_params = {}
    #augmenter_params["cutout_const"] = None
    #augmenter_params["translate_const"] = None
    #augmenter_params["num_layers"] = None
    #augmenter_params["magnitude"] = None
    #augmenter_params["autoaugmentation_name"] = None

    builders = []
    builders.append(dataset_factory.Dataset(
        dataset=dataset,
        index_file_dir=None,
        split="train",
        image_size=dim[0],
        num_classes=n_classes,
        num_channels=3,
        batch_size=batch_size,
        dtype='float32',
        one_hot=True,
        use_dali=False,
        augmenter=augmenter,
        cache=False,
        mean_subtract=False,
        standardize=False,
        augmenter_params=augmenter_params,
        cutmix_alpha=cutmix_alpha, 
        mixup_alpha=mixup_alpha,
        defer_img_mixing=True,
        model_preprocess_func=lambda x:model_handler.preprocess_func(x, None),
        disable_map_parallelization=False))

    val_split = "test"
    if dataset == "imagenet2012":
        val_split = "validation"

    builders.append(dataset_factory.Dataset(
        dataset=dataset,
        index_file_dir=None,
        split=val_split,
        image_size=dim[0],
        num_classes=n_classes,
        num_channels=3,
        batch_size=batch_size,
        dtype='float32',
        one_hot=True,
        use_dali=False,
        augmenter=None,
        cache=False,
        mean_subtract=False,
        standardize=False,
        augmenter_params=None,
        cutmix_alpha=0.0, 
        mixup_alpha=0.0,
        defer_img_mixing=False,
        hvd_size=hvd.size(),
        model_preprocess_func=lambda x:model_handler.preprocess_func(x, None),
        disable_map_parallelization=False))

    return [ builder.build() for builder in builders ]

def load_dataset(dataset, model_handler, sampling_ratio=1.0, training_augment=True, n_classes=100):

    batch_size = model_handler.get_batch_size(dataset)

    if dataset in ["imagenet2012", "cifar100"]:
        train_data_generator, valid_data_generator = load_data_nvidia(dataset, model_handler, sampling_ratio=sampling_ratio, training_augment=training_augment, n_classes=n_classes)

        if dataset == "imagenet2012": 
            num_train_examples = 1281167
            num_val_examples = 50000
        else:
            num_train_examples = 50000
            num_val_examples = 10000
        iters = num_train_examples // (batch_size * hvd.size())
        iters_val = num_val_examples // (batch_size * hvd.size())
        test_data_generator = valid_data_generator

    else:
        train_data_generator, valid_data_generator, test_data_generator = load_data(dataset, model_handler, sampling_ratio=sampling_ratio, training_augment=training_augment, n_classes=n_classes)
        iters = len(train_data_generator)
        iters_val = len(valid_data_generator)

    return (train_data_generator, valid_data_generator, test_data_generator), (iters, iters_val)

def train(
    dataset,
    model,
    model_name,
    model_handler,
    epochs,
    callbacks=None,
    sampling_ratio=1.0,
    augment=True,
    exclude_val=False,
    n_classes=100,
    save_dir=None,
    conf=None):

    batch_size = model_handler.get_batch_size(dataset)

    if type(dataset) == str:
        data_gen, iters_info = load_dataset(dataset, model_handler, sampling_ratio=sampling_ratio, training_augment=augment, n_classes=n_classes)
    else:
        data_gen, iters_info = dataset
    train_data_generator, valid_data_generator, test_data_generator = data_gen
    iters, iters_val = iters_info
    iters = int(iters * sampling_ratio)

    if callbacks is None:
        callbacks = []

    if conf is not None:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())

        lr_params = {
            'name':conf["lr_name"],
            'initial_lr': conf["initial_lr"],
            'decay_epochs': conf["decay_epochs"],
            'decay_rate': conf["decay_rate"],
            'warmup_epochs': conf["warmup_epochs"],
            'examples_per_epoch': None,
            'boundaries': None,
            'multipliers': None,
            'scale_by_batch_size': 1./float(batch_size),
            'staircase': True,
            't_mul': conf["t_mul"],
            'm_mul': conf["m_mul"],
            'alpha': conf['alpha']
        }

        learning_rate = optimizer_factory.build_learning_rate(
            params=lr_params,
            batch_size=batch_size * hvd.size() * conf["grad_accum_steps"], # updates are iteration based not batch-index based
            train_steps=iters,
            max_epochs=epochs)

        opt_params = {
            'name': conf["opt_name"],
            'decay': conf["decay"],
            'epsilon': conf["epsilon"],
            'momentum': conf["momentum"],
            'lookahead': conf["lookahead"],
            'moving_average_decay': conf["moving_average_decay"],
            'nesterov': conf["nesterov"],
            'beta_1': conf["beta_1"],
            'beta_2': conf["beta_2"],

        }
        
        # set up optimizer
        optimizer = optimizer_factory.build_optimizer(
            optimizer_name=conf["opt_name"],
            base_learning_rate=learning_rate,
            params=opt_params
        )

        if conf["use_amp"] and conf["grad_accum_steps"] > 1:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        elif conf["grad_accum_steps"] == 1:
            optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.fp16 if conf["hvd_fp16_compression"] else hvd.Compression.none)

        # compile
        if conf["mode"] in ["transfer_learning", "distillation"]:
            loss = {model.output[0].name.split("/")[0]:BespokeTaskLoss()}
            metrics={model.output[0].name.split("/")[0]:accuracy}
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False, experimental_run_tf_function=False)
        else:
            model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'], run_eagerly=False, experimental_run_tf_function=False)

        if conf["moving_average_decay"] > 0:
            callbacks.append(
                custom_callbacks.MovingAverageCallback(intratrain_eval_using_ema=conf["intratrain_eval_using_ema"]))

    else:
        # model was already compiled
        pass

    if save_dir is not None and hvd.local_rank() == 0:
        model_name_ = '%s_model.{epoch:03d}.h5' % (model_name+"_"+dataset)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name_)

        if conf is not None and conf["moving_average_decay"] > 0:
            mchk = custom_callbacks.AverageModelCheckpoint(update_weights=False,
                                          filepath=filepath,
                                          verbose=0,
                                          save_best_only=True,
                                          save_weights_only=False,
                                          mode="auto",
                                          save_freq="epoch")
        else:
            mchk = keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor="val_accuracy",
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode="auto",
                save_freq="epoch",
                options=None,
            )
        callbacks.append(mchk)

    if exclude_val:
        model_history = model.fit(train_data_generator,
                                        callbacks=callbacks,
                                        verbose=1 if hvd.local_rank() == 0 else 0,
                                        epochs=epochs,
                                        steps_per_epoch=iters * conf["grad_accum_steps"])
    else:
        model_history = model.fit(train_data_generator,
                                        validation_data=valid_data_generator,
                                        callbacks=callbacks,
                                        verbose=1 if hvd.local_rank() == 0 else 0,
                                        epochs=epochs,
                                        steps_per_epoch=iters * conf["grad_accum_steps"],
                                        validation_steps=iters_val)

    del train_data_generator, valid_data_generator
