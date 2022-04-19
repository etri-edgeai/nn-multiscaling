from __future__ import absolute_import
from __future__ import print_function

from bespoke import backend as B
from bespoke.base.topology import AlternativeNode
from bespoke.generator import *

class ModelHouse(object):
    """This is a class of housing a model for model scaling.

    """

    def __init__(self, model, custom_objects=None):
        self._namespace = set()
        model_, parser = B.preprocess(model, self._namespace, custom_objects)
        self._model = model_
        self._parser = parser
        self._root = None
        self._custom_objects = custom_objects

        self.build()

    def build(self):
        self._root = self._parser.build()

        gen_ = PruningGenerator(self._namespace)

        # traverse and generate alternative
        def make_alternatives(node):
            alternatives = gen_.generate(node.net, custom_objects=self._custom_objects)
            for idx, a in enumerate(alternatives):
                node.add(node.id_ +"_"+ str(idx), a)

        stk = [self._root]
        while len(stk) > 0:
            curr = stk.pop()
            for child in curr.children:
                stk.append(child.node)
            make_alternatives(curr)

    def make_train_model(self, range_=None):
        return B.make_train_model(self._root, range_=range_)

    def extract(self, recipe):
         

        pass

    def query(self):
        pass

    def profile(self):
        pass

    def dump(self):
        pass
    
    def load(self):
        pass

    def add(self):
        pass

    def remove(self):
        pass


def test():

    from tensorflow import keras
    import tensorflow as tf
    import numpy as np

    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(32,32,3), classes=100)

    tf.keras.utils.plot_model(model, to_file="original.png", show_shapes=True)

    mh = ModelHouse(model)

    # random data test
    data = np.random.rand(1,32,32,3)
    house = mh.make_train_model()

    tf.keras.utils.plot_model(house, to_file="house.pdf", show_shapes=True)
    y = house(data)






    mh.query(0.5)

   


if __name__ == "__main__":
    test()
