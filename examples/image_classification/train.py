import math
import os
import logging

from tqdm import tqdm
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from datagen_ds import DataGenerator
from dataloader import dataset_factory

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

def load_data_dali(dataset, model_handler, sampling_ratio=1.0, training_augment=True, batch_size=-1, n_classes=100):

    builders = []
    validation_dataset_builder = None
    train_dataset_builder = None

    mode = "train_eval" 
    if "train" in mode:
        img_size = 224
        print("Image size {} used for training".format(img_size))
        print("Train batch size {}".format(32))
        train_dataset_builder = dataset_factory.Dataset(data_dir="/ssd_data/tensorflow_datasets_/imagenet2012/5.1.0_dali",
        index_file_dir="/ssd_data/tensorflow_datasets_/imagenet2012/5.1.0_dali_index",
        split='train',
        num_classes=n_classes,
        image_size=img_size,
        batch_size=32,
        one_hot=True,
        use_dali=True,
        augmenter=None,
        #augmenter_params=build_augmenter_params(params.augmenter_name, 
        #    params.cutout_const, 
        #    params.translate_const, 
        #    params.raug_num_layers, 
        #    params.raug_magnitude, 
        #    params.autoaugmentation_name),
        mixup_alpha=0.5,
        cutmix_alpha=0.5,
        defer_img_mixing=True,
        mean_subtract=False,
        standardize=False,
        hvd_size=None,
        disable_map_parallelization=False
        )
    if "eval" in mode:
        img_size = 224
        print("Image size {} used for evaluation".format(img_size))
        validation_dataset_builder = dataset_factory.Dataset(data_dir="/ssd_data/tensorflow_datasets_/imagenet2012/5.1.0_dali",
        index_file_dir="/ssd_data/tensorflow_datasets_/imagenet2012/5.1.0_dali_index",
        split='validation',
        num_classes=n_classes,
        image_size=img_size,
        batch_size=32,
        one_hot=False,
        use_dali=True,
        hvd_size=None)

    builders.append(train_dataset_builder)
    builders.append(validation_dataset_builder)
    datasets = [builder.build() if builder else None for builder in builders]
    return datasets


def train(dataset, model, model_name, model_handler, epochs, sampling_ratio=1.0, callbacks=None, augment=True, exclude_val=False, n_classes=100, save_dir=None, use_dali=False):

    if use_dali:
        assert dataset == "imagenet2012"
        train_data_generator, valid_data_generator = load_data_dali(dataset, model_handler, sampling_ratio=sampling_ratio, training_augment=augment, n_classes=n_classes)
        iters = 40023
        iters_val = 1000
    else:
        train_data_generator, valid_data_generator, test_data_generator = load_data(dataset, model_handler, sampling_ratio=sampling_ratio, training_augment=augment, n_classes=n_classes)
        iters = len(train_data_generator)
        iters_val = len(valid_data_generator)

    if callbacks is None:   
        callbacks = []

    callbacks_ = model_handler.get_callbacks(iters)
    callbacks += callbacks_

    # Prepare model model saving directory.
    if save_dir is not None:
        model_name_ = '%s_model.{epoch:03d}.h5' % (model_name+"_"+dataset)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name_)

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
                                        verbose=1,
                                        epochs=epochs,
                                        steps_per_epoch=iters)
    else:
        model_history = model.fit(train_data_generator,
                                        validation_data=valid_data_generator,
                                        callbacks=callbacks,
                                        verbose=1,
                                        epochs=epochs,
                                        steps_per_epoch=iters,
                                        validation_steps=iters_val)

    del train_data_generator, valid_data_generator, test_data_generator
