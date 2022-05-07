import os
import copy
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from orderedset import OrderedSet

from nncompress.backend.tensorflow_.transformation import handler
from bespoke.base.interface import ModelHouse
from bespoke.base.builder import RandomHouseBuilder

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#import resnet50 as model_handler
from models import efficientnet as model_handler
from train import train, iteration_based_train, load_data

dataset = "cifar100"

#model = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling=None, classes=10)
#model = tf.keras.applications.DenseNet121(include_top=False, weights=None, pooling=None, classes=10)
#model = model_handler.get_model("cifar100")
model = tf.keras.models.load_model("pretrained_weights/efnet_cifar100_model.059.h5")

#model_handler.compile(model, run_eagerly=True, loss="categorical_crossentropy")
#train(dataset, model, "test", model_handler, 5, callbacks=None, augment=True, exclude_val=False, n_classes=100)

tf.keras.utils.plot_model(model, to_file="original.png", show_shapes=True)

mh = ModelHouse(model)
b = RandomHouseBuilder(mh)
b.build(50)
mh.build_base(memory_limit=205168896)
mh.save("saved_efnet_house_1")

for n in mh.nodes:
   n.sleep() # to_cpu 
tf.keras.backend.clear_session()

print("after build")
print(tf.config.experimental.get_memory_info("GPU:0"))

trainable_nodes = [n for n in mh.trainable_nodes]
CNT = 20
to_remove = []
while len(trainable_nodes) > 0:
    cnt = CNT
    while True:
        if cnt == 0:
            to_remove.append(trainable_nodes.pop())
            cnt = CNT

        tf.keras.backend.clear_session()
        print("Now training ... %d" % len(trainable_nodes))

        try:
            targets = []
            for i in range(len(trainable_nodes)):
                if cnt == i:
                    break
                targets.append(trainable_nodes[i])
                targets[-1].wakeup()

        except Exception as e:
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
            tf.keras.utils.plot_model(house, "hhhlarge.pdf", expand_nested=True)
            model_handler.compile(house, run_eagerly=True)
            print(tf.config.experimental.get_memory_info("GPU:0"))

            train(dataset, house, "test", model_handler, 3, callbacks=None, augment=True, n_classes=100)
            del house, output_idx, output_map
            import gc
            gc.collect()

        except Exception as e:
            print(e)
            print("Memory Problem Occurs %d" % cnt)
            tf.keras.utils.plot_model(house, "hhhlarge.pdf", expand_nested=True)
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

mh.save("saved_efnet_house")
DONE

train_data_generator, _, _ = load_data(dataset, model_handler, training_augment=True, n_classes=100)
sample_inputs = []
for x,y in train_data_generator:
    sample_inputs.append(x)
    if len(sample_inputs) > 30:
        break

#mh = ModelHouse(model=None)
#mh.load("ttt")
mh.build_sample_data(sample_inputs)
mh.profile()

# check mh and mh2 are same.
#assert mh._namespace == mh2._namespace
#for n1, n2 in zip(mh._nodes, mh2._nodes):
#    assert (n1.net.model.get_weights()[0] == n2.net.model.get_weights()[0]).all()


#data = np.random.rand(1,224,224,3)
#y = house(data)
#tf.keras.utils.plot_model(house, to_file="house.pdf", show_shapes=True)


#iteration_based_train(dataset, house, model_handler, 500, output_idx, output_map, lr_mode=0, stopping_callback=None, augment=True, n_classes=100, eval_steps=-1, validate_func=None)
#mh.extract({5: 0, 1: 1})
