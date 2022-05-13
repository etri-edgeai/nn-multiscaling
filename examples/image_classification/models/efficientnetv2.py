
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization
import numpy as np
import cv2

from .loss import BespokeTaskLoss, accuracy

height = 224
width = 224
input_shape = (height, width, 3) # network input
batch_size = 32

def get_shape(dataset):
    return (height, width, 3) # network input

def get_batch_size(dataset):
    return batch_size

def get_name():
    return "efnetv2b0"

def preprocess_func(img, shape):
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return img

def parse_fn(example_serialized):
    image = example_serialized["image"]
    resized_image = tf.keras.preprocessing.image.smart_resize(image, [256, 256], interpolation='bicubic')
    image = _central_crop([image], 224, 224)[0]
    return {"image": image, "label":example_serialized["label"]}

def get_model(dataset, n_classes=100):
    if dataset == "imagenet2012":
        model = tf.keras.applications.efficientnet_v2.EfficientNetB0(
            include_top=True, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling=None, classes=1000,
            classifier_activation='softmax')
        return model
    else:
        efnb0 = tf.keras.applications.efficientnet_v2.EfficientNetB0(
            include_top=False, weights='imagenet', input_shape=input_shape, classes=n_classes)

        model = Sequential()
        model.add(efnb0)
        model.add(GlobalAveragePooling2D())
        if dataset == "cifar100":
            model.add(Dropout(0.5))
        else:
            model.add(Dropout(0.25))
        model.add(Dense(n_classes, activation='softmax'))
        return model

def get_optimizer(mode=0):
    if mode == 0:
        return Adam(lr=0.0001)
    elif mode == 1:
        return Adam(lr=0.00001)


def compile(model, run_eagerly=True, loss={'dense':BespokeTaskLoss()}, metrics={'dense':accuracy}, transfer=False):
    optimizer = Adam(lr=0.0001)
    if transfer:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly)
    else:
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'], run_eagerly=run_eagerly)

def get_callbacks(nsteps=0):
    #early stopping to monitor the validation loss and avoid overfitting
    #early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

    #reducing learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    return [rlrop]

def get_custom_objects():
    return None

def get_train_epochs(finetune=False):
    if finetune:
        return 50
    else:
        return 100

def fix_mean_variance():
    return tf.convert_to_tensor([[[[0.485, 0.456, 0.406]]]]), tf.convert_to_tensor([[[[0.229, 0.224, 0.225]]]])

def get_heuristic_positions():

    return [
        "block1a_project_bn",
        "block2b_add",
        "block3b_add",
        "block4c_add",
        "block5c_add",
        "block6d_add",
        "top_activation"
    ]
