""" TaskBuilder for Image Classfication

"""
import tensorflow as tf
from tensorflow.keras import mixed_precision

from bespoke.base.task import TaskBuilder
from bespoke.train.utils import optimizer_factory
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import\
    PruningNNParser, StopGradientLayer, has_intersection

import prep
from dataset_loader import load_dataset

def make_tl_dataset(dataset):
    """ make tl dataset """
    # do nothing
    return dataset

class ImageClassificationBuilder(TaskBuilder):
    """ Task Builder

    """

    def __init__(self, config):
        """ Init function """
        self.config = config

    def load_dataset(self, split=None, is_tl=False):
        """Implement your dataloader

        """ 
        (train_gen, val_gen, test_gen), (iters, iters_val) = load_dataset(
            self.config["dataset"],
            self.config["width"],
            self.config["batch_size"],
            training_augment=self.config["augment"],
            n_classes=self.config["n_classes"],
            sampling_ratio=self.config["sampling_ratio"],
            cutmix_alpha=self.config["cutmix_alpha"],
            mixup_alpha=self.config["mixup_alpha"])

        if is_tl:
            train_gen = make_tl_dataset(train_gen)
            val_gen = make_tl_dataset(val_gen)
            test_gen = make_tl_dataset(test_gen)

        if split == "train":
            return train_gen
        elif split == "val":
            return val_gen
        elif split == "test":
            return test_gen
        else:
            return (train_gen, val_gen, test_gen), (iters, iters_val)

    def prep(self, model, is_teacher=False, for_benchmark=False):
        """ Preparation of your model

        """

        if self.config["use_amp"]:
            model = prep.change_dtype(
                model, mixed_precision.global_policy(), custom_objects=self.get_custom_objects())
        if is_teacher or for_benchmark:
            model = prep.remove_augmentation(model, custom_objects=self.get_custom_objects())
        else:
            model = prep.add_augmentation(
                model,
                self.config["width"],
                train_batch_size=self.config["batch_size"],
                do_mixup=self.config["mixup_alpha"] > 0,
                do_cutmix=self.config["cutmix_alpha"] > 0,
                custom_objects=self.get_custom_objects(),
                update_batch_size=True)

        return model

    def compile(self, model, mode="eval", run_eagerly=True):
        """ Compile your model

        """
        if mode == "eval":
            adam = tf.keras.optimizers.Adam(0.001)
            model.compile(
                optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'], run_eagerly=run_eagerly)
        else:
            raise NotImplementedError("Not implemented")
