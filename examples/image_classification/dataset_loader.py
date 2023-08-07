import math
import os
import logging

from tqdm import tqdm
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import horovod.tensorflow.keras as hvd

from dataloader import dataset_factory

def center_crop_and_resize(image, image_size, crop_padding=32, interpolation='bicubic'):
    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]

    padded_center_crop_size =\
        tf.cast((image_size / (image_size + crop_padding)) * tf.cast(tf.math.minimum(h, w), tf.float32), tf.int32)
    offset_height = ((h - padded_center_crop_size) + 1) // 2
    offset_width = ((w - padded_center_crop_size) + 1) // 2

    image_crop = image[offset_height:padded_center_crop_size + offset_height,
                       offset_width:padded_center_crop_size + offset_width]

    resized_image = tf.keras.preprocessing.image.smart_resize(
        image, [image_size, image_size], interpolation=interpolation)
    return resized_image

def data_preprocess_func(img, width):
    img = center_crop_and_resize(img, width)
    #img = preprocess_input(img)
    return img

def model_preprocess_func(img, shape):
    img = tf.keras.applications.imagenet_utils.preprocess_input(
        img, data_format=None, mode='torch'
        )
    #img = preprocess_input(img)
    return img

def load_data_nvidia(
    dataset,
    width,
    training_augment=True,
    batch_size=-1,
    n_classes=100,
    cutmix_alpha=0.0,
    mixup_alpha=0.0,
    sampling_count=None):

    if sampling_count is None:
        sampling_count = (None, None)

    if dataset == "imagenet2012":
        data_dir = "tensorflow_datasets/imagenet2012/5.1.0_dali"
    elif dataset == "cifar100":
        data_dir = "tensorflow_datasets/cifar100/3.0.2_dali"
    else:
        raise NotImplementedError("no support for the other datasets")

    dim = (width, width)

    augmenter = "autoaugment"
    augmenter = None
    augmenter_params = {}

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
        sampling_count=sampling_count[0],
        data_preprocess_func=lambda x:data_preprocess_func(x, width),
        model_preprocess_func=lambda x:model_preprocess_func(x, None),
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
        sampling_count=sampling_count[1],
        hvd_size=hvd.size(),
        data_preprocess_func=lambda x:data_preprocess_func(x, width),
        model_preprocess_func=lambda x:model_preprocess_func(x, None),
        disable_map_parallelization=False))

    return [ builder.build() for builder in builders ]

def load_dataset(
    dataset,
    width,
    batch_size,
    training_augment=True,
    n_classes=100,
    sampling_ratio=1.0,
    cutmix_alpha=0.5,
    mixup_alpha=0.5):

    if dataset == "imagenet2012": 
        num_train_examples = 1281167
        num_val_examples = 50000
    elif dataset == "cifar100":
        num_train_examples = 50000
        num_val_examples = 10000
    elif dataset == "caltech_birds2011":
        num_train_examples = 5994
        num_val_examples = 5794
    else:
        num_train_examples = 3680
        num_val_examples = 3669

    if sampling_ratio != 1.0:
        sampling_count = (int(num_train_examples * sampling_ratio), int(num_val_examples * sampling_ratio))
    else:
        sampling_count = (None, None)

    train_data_generator, valid_data_generator = load_data_nvidia(
        dataset,
        width=width,
        batch_size=batch_size,
        training_augment=training_augment,
        n_classes=n_classes,
        sampling_count=sampling_count,
        cutmix_alpha=cutmix_alpha,
        mixup_alpha=mixup_alpha)

    iters = num_train_examples // (batch_size * hvd.size())
    iters_val = num_val_examples // (batch_size * hvd.size())
    test_data_generator = valid_data_generator

    return (train_data_generator, valid_data_generator, test_data_generator), (iters, iters_val)
