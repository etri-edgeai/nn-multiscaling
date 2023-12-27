""" TaskBuilder Base

"""

import tensorflow as tf

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer

from bespoke.train.utils import optimizer_factory

class TaskBuilder(object):
    """ Task Builder

    """

    def __init__(self, config):
        self.config = config

    def get_optimizer_gen(self, config, is_tl=False, is_distil=False):
        """Define a python function to get your optimizer

        """
        return None

    def get_loss_gen(self, config, is_tl=False, is_distil=False):
        """Define a python function to get your task loss

        """
        return None

    def get_callbacks_gen(self, config, is_tl=False, is_distil=False):
        """Define a python function to get your training callbacks

        """
        return None

    def get_metrics_gen(self, config, is_tl=False, is_distil=False):
        """Define a python function to get your training metrics

        """
        return None

    def load_dataset(self, split=None, is_tl=False):
        """Implement your dataloader

        """ 
        return None

    def get_custom_objects_imple(self):
        """ Custom objects dict

        """
        return {}

    def load_model(self, model_path):
        """ Load model

        """
        model = tf.keras.models.load_model(model_path, custom_objects=self.get_custom_objects())
        return model

    def prep(self, model, is_teacher=False, for_benchmark=False):
        """ Preparation of your model

        """
        return None

    def compile(self, model, mode="eval"):
        """ Compile your model

        """
        pass

    def get_custom_objects(self):
        """ Custom objects dict

        """
        custom_objects = {
            "SimplePruningGate":SimplePruningGate,
            "StopGradientLayer":StopGradientLayer,
            "HvdMovingAverage":optimizer_factory.HvdMovingAverage
        }
        custom_objects.update(self.get_custom_objects_imple())
        return custom_objects
