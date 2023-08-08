""" Train wrapper

"""

import math
import os
import logging

from tqdm import tqdm
from tensorflow import keras
import tensorflow as tf
import horovod.tensorflow.keras as hvd

from bespoke.train.utils import callbacks as custom_callbacks
from bespoke.train.utils import optimizer_factory

def train(
    config,
    model,
    epochs,
    batch_size,
    data_gen, # augmentation is already applied
    is_tl=False,
    is_distil=False,
    get_optimizer_gen=None,
    get_loss_gen=None,
    get_callbacks_gen=None,
    get_metrics_gen=None,
    prefix="model",
    exclude_val=False,
    sampling_ratio=1.0,
    save_dir=None):
    """ train function

    """

    (train_data_generator, valid_data_generator, test_data_generator), (iters, iters_val) = data_gen

    get_optimizer_gen = default_get_optimizer_gen if get_optimizer_gen(config) is None else get_optimizer_gen
    get_loss_gen = default_get_loss_gen if get_loss_gen(config) is None else get_loss_gen
    get_callbacks_gen = default_get_callbacks_gen if get_callbacks_gen(config) is None else get_callbacks_gen
    get_metrics_gen = default_get_metrics_gen if get_metrics_gen(config) is None else get_metrics_gen

    if exclude_val:
        iters_val = None
    iters = int(iters * sampling_ratio)

    optimizer, learning_rate = get_optimizer_gen(config, is_tl=is_tl, is_distil=is_distil)(iters, epochs)
    callbacks = get_callbacks_gen(config, is_tl=is_tl, is_distil=is_distil)(model, optimizer, learning_rate)
    loss = get_loss_gen(config, is_tl=is_tl, is_distil=is_distil)(model)
    metrics, monitor = get_metrics_gen(config, is_tl=is_tl, is_distil=is_distil)(model)

    # compile
    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False, experimental_run_tf_function=False)

    if save_dir is not None and hvd.local_rank() == 0:
        model_name_ = '%s.best.h5' % prefix
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name_)

        if config["moving_average_decay"] > 0:
            mchk = custom_callbacks.AverageModelCheckpoint(update_weights=False,
                                          filepath=filepath,
                                          monitor=monitor,
                                          verbose=0,
                                          save_best_only=True,
                                          save_weights_only=False,
                                          mode="auto",
                                          save_freq="epoch")
            func = model.save
            model.save = lambda filepath, overwrite=True, options=None:\
                func(filepath, overwrite=overwrite, include_optimizer=False, options=options)

        else:
            mchk = keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor=monitor,
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode="auto",
                save_freq="epoch",
                options=None,
            )
        callbacks.append(mchk)

    model_history = model.fit(train_data_generator,
                                    validation_data=valid_data_generator,
                                    callbacks=callbacks,
                                    verbose=1 if hvd.local_rank() == 0 else 0,
                                    epochs=epochs,
                                    steps_per_epoch=iters,
                                    validation_steps=iters_val)

    del train_data_generator, valid_data_generator
    return model

def default_get_optimizer_gen(config, is_tl=False, is_distil=False):
    """ Default get_optimizer_gen

    """

    def get_optimizer(iters, epochs):

        lr_params = {
            'name':config["lr_name"],
            'initial_lr': config["initial_lr"],
            'decay_epochs': config["decay_epochs"],
            'decay_rate': config["decay_rate"],
            'warmup_epochs': config["warmup_epochs"],
            'examples_per_epoch': None,
            'boundaries': None,
            'multipliers': None,
            'scale_by_batch_size': 0.0,
            'staircase': True,
            't_mul': config["t_mul"],
            'm_mul': config["m_mul"],
            'alpha': config['alpha']
        }

        learning_rate = optimizer_factory.build_learning_rate(
            params=lr_params,
            batch_size=config["batch_size"] * hvd.size(),
            train_steps=iters,
            max_epochs=epochs)

        opt_params = {
            'name': config["opt_name"],
            'decay': config["decay"],
            'epsilon': config["epsilon"],
            'momentum': config["momentum"],
            'lookahead': config["lookahead"],
            'moving_average_decay': config["moving_average_decay"],
            'nesterov': config["nesterov"],
            'beta_1': config["beta_1"],
            'beta_2': config["beta_2"],
        }
     
        # set up optimizer
        optimizer = optimizer_factory.build_optimizer(
            optimizer_name=config["opt_name"],
            base_learning_rate=learning_rate,
            params=opt_params
        )

        optimizer = hvd.DistributedOptimizer(
            optimizer, compression=hvd.Compression.fp16 if config["hvd_fp16_compression"] else hvd.Compression.none)

        return optimizer, learning_rate
    return get_optimizer


def default_get_loss_gen(config, is_tl=False, is_distil=False):
    """ Default get_loss_gen

    """
    from bespoke.train.utils import misc
    def get_loss(model):
        loss = {model.output[0].name.split("/")[0]:misc.BespokeTaskLoss(label_smoothing=config["label_smoothing"])}
        return loss
    
    def get_loss_finetune(model):
        return "categorical_crossentropy"

    if is_tl or is_distil:
        return get_loss
    else:
        return get_loss_finetune

def default_get_callbacks_gen(config, is_tl=False, is_distil=False):
    """ Default get_callbacks_gen

    """
    def get_callbacks(model, optimizer, learning_rate):
        callbacks = []
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())

        class StepCounter(tf.keras.callbacks.Callback):
            """ Compute num of steps """

            def __init__(self, scheduler):
                """ Init function """
                super(StepCounter, self).__init__()
                self.scheduler = scheduler
                self._counter = 1

            def on_train_batch_begin(self, batch, logs=None):
                """ on train batch begin """
                self._counter += 1

            def on_epoch_begin(self, epoch, logs=None):
                """ on epoch begin """
                print(self.scheduler(self._counter))

        if hvd.rank() == 0:
            counter_ = StepCounter(learning_rate)
            callbacks.append(counter_)

        if config["moving_average_decay"] > 0:
            callbacks.append(
                custom_callbacks.MovingAverageCallback(intratrain_eval_using_ema=config["intratrain_eval_using_ema"]))

        return callbacks
    return get_callbacks

def default_get_metrics_gen(config, is_tl=False, is_distil=False):
    """ Default get_metrics_gen

    """
    from bespoke.train.utils import misc as misc
    def get_metrics(model):
        """ get metrics """
        metrics={model.output[0].name.split("/")[0]:misc.accuracy}
        return metrics, "val_accuracy"

    def get_metrics_finetune(model):
        """ get metrics for finetuning """
        return "accuracy", "val_accuracy"
    
    if is_tl or is_distil:
        return get_metrics
    else:
        return get_metrics_finetune
