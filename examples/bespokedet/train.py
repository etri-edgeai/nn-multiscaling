import os
import copy

import horovod.tensorflow.keras as hvd
from automl.efficientdet.tf2.train import setup_model
import tensorflow as tf
from automl.efficientdet.tf2 import train_lib

def setup_optimizer(
    config):
    """`train_lib.get_optimizer` caller

        Args.

            optimizer: str, the optimizer name
            momentum: float, a momentum value
            moving_average_decay: float, moving average decay value
            mixed_precision: str, mixed_float16, ...
            loss_scale: float, loss scale value

        Returns.
            An optimizer instance

    """

    optimizer = train_lib.get_optimizer(copy.deepcopy(config))

    if config["use_hvd"]:
        optimizer = hvd.DistributedOptimizer(
            optimizer, compression=hvd.Compression.fp16 if config["mixed_precision"] else hvd.Compression.none)

    return optimizer


def setup_model(
    model,
    optimizer,
    image_size,
    delta,
    iou_loss_type,
    min_level,
    max_level,
    num_scales,
    aspect_ratios,
    anchor_scale,
    alpha,
    gamma,
    use_distillation,
    label_smoothing=0.1):
    """Reimplementation of `train_lib.setup_model` (Inplace)

        Args.

            model:
            optimizer:
            image_size:
            delta:
            iou_loss_type:
            min_level:
            max_level:
            num_scales:
            aspect_ratio
            anchor_scale:
            alpha:
            gamma:
            label_smoothing:

    """
    model.build((None, *image_size, 3))
    model.compile(
        optimizer=optimizer,
        loss={
            train_lib.BoxLoss.__name__:
                train_lib.BoxLoss(
                    delta, reduction=tf.keras.losses.Reduction.NONE),
            train_lib.BoxIouLoss.__name__:
                train_lib.BoxIouLoss(
                    iou_loss_type,
                    min_level,
                    max_level,
                    num_scales,
                    aspect_ratios,
                    anchor_scale,
                    image_size,
                    reduction=tf.keras.losses.Reduction.NONE),
            train_lib.FocalLoss.__name__:
                train_lib.FocalLoss(
                    alpha,
                    gamma,
                    label_smoothing=label_smoothing,
                    reduction=tf.keras.losses.Reduction.NONE),
            tf.keras.losses.SparseCategoricalCrossentropy.__name__:
                tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        }, run_eagerly=False)

def train(
    model,
    train_dataset,
    val_dataset,
    num_iters,
    num_iters_val,
    epochs,
    prefix=None,
    save_dir=None,
    callbacks=None,
    use_hvd=False):
    """Train function

        Args.

            model: keras.Model, a model to train
            train_dataset: tf.dataset, a training dataset
            val_dataset: tf.dataset, a validation dataset
            num_iters: int, the number of iterations per training epoch
            num_iters_val: int, the number of iterations per validation epoch
            save_dir: str, the path for saving model weights
            callbacks: list, a list of keras.Callback instances
            use_hvd: bool, use horovod.
            use_amp: bool, use mixed_preicision

    """

    if callbacks is None or (use_hvd and hvd_local_rank() != 0):
        callbacks = []

    if use_hvd:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())

    if save_dir is not None and hvd.local_rank() == 0:
        name_ = '%s_model.best.h5' % (prefix)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, name_)

    model_history = model.fit(train_dataset,
                              validation_data=val_dataset,
                              callbacks=callbacks,
                              initial_epoch=model.optimizer.iterations.numpy() // num_iters,
                              verbose=1 if hvd.local_rank() == 0 else 0,
                              epochs=epochs,
                              steps_per_epoch=num_iters,
                              validation_steps=num_iters_val)
