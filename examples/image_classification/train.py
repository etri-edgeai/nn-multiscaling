import math
import os
import logging

from tqdm import tqdm
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from datagen_ds import DataGenerator

def load_data(dataset, model_handler, sampling_ratio=1.0, training_augment=True, batch_size=-1, n_classes=100):

    dim = (224, 224)
    preprocess_func = model_handler.preprocess_func
    if hasattr(model_handler, "batch_preprocess_func"):
        batch_pf = model_handler.batch_preprocess_func
    else:
        batch_pf = None

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
        sampling_ratio=sampling_ratio)

    valid_data_generator = DataGenerator(
        ds_val,
        dataset=dataset,
        batch_size=batch_size_,
        augment=False,
        dim=dim,
        n_classes=n_classes,
        n_examples=val_examples,
        preprocess_func=preprocess_func,
        is_batched=is_batched,
        batch_preprocess_func=batch_pf)

    test_data_generator = DataGenerator(
        ds_val,
        dataset=dataset,
        batch_size=batch_size_,
        augment=False,
        dim=dim,
        n_classes=n_classes,
        n_examples=val_examples,
        preprocess_func=preprocess_func,
        is_batched=is_batched,
        batch_preprocess_func=batch_pf)

    return train_data_generator, valid_data_generator, test_data_generator


def train_step(X, model, output_idx, output_map, y):
    with tf.GradientTape() as tape:
        logits = model(X)
        if type(logits) != list:
            logits = [logits]

        loss = tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(logits[0], y))
        if len(output_map) > 0:
            temp = None
            for tname, sname in output_map: # exclude the model's ouptut logit.
                t = logits[output_idx[tname]]
                s = logits[output_idx[sname]]
                if temp is None:
                    temp = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(t, s))
                else:
                    temp += tf.math.reduce_mean(tf.keras.losses.mean_squared_error(t, s))
            temp /= len(output_map)
        loss += temp
    return tape, loss


def iteration_based_train(dataset, model, model_handler, max_iters, output_idx, output_map, sampling_ratio=1.0, lr_mode=0, stopping_callback=None, augment=True, n_classes=100, eval_steps=-1, validate_func=None):

    train_data_generator, valid_data_generator, test_data_generator = load_data(dataset, model_handler, sampling_ratio=sampling_ratio, training_augment=augment, n_classes=n_classes)

    if dataset == "imagenet":
        iters = int(math.ceil(1281167.0 / model_handler.batch_size))
    else:
        iters = len(train_data_generator)

    global_step = 0
    #callbacks_ = model_handler.get_callbacks(iters)
    optimizer = model_handler.get_optimizer(lr_mode)

    epoch = 0
    with tqdm(total=max_iters, ncols=120) as pbar:
        while global_step < max_iters: 
            # start with new epoch.
            done = False
            idx = 0
            for X, y in train_data_generator:
                idx += 1
                y = tf.convert_to_tensor(y, dtype=tf.float32)

                tape, loss = train_step(X, model, output_idx, output_map, y)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                global_step += 1
                pbar.update(1)
                #print(" / ", tf.config.experimental.get_memory_info('GPU:0'))

                if eval_steps != -1 and global_step % eval_steps == 0 and validate_func is not None:
                    val = validate_func()
                    print("Global Steps %d: %f" % (global_step, val))
                    logging.info("Global Steps %d: %f" % (global_step, val))

                if stopping_callback is not None and stopping_callback(idx, global_step):
                    done = True
                    break
            if done:
                break
            else:
                train_data_generator.on_epoch_end()

            #epoch += 1
            #if validate_func is not None:
            #    print("Epoch %d: %f" % (epoch, validate_func()))



def train(dataset, model, model_name, model_handler, epochs, sampling_ratio=1.0, callbacks=None, augment=True, exclude_val=False, n_classes=100, save_dir=None):

    train_data_generator, valid_data_generator, test_data_generator = load_data(dataset, model_handler, sampling_ratio=sampling_ratio, training_augment=augment, n_classes=n_classes)

    if callbacks is None:   
        callbacks = []

    iters = len(train_data_generator)

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
        #callbacks.append(mchk)

    """
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = "logs",
        histogram_freq = 1
    )
    callbacks.append(tensorboard_callback)
    """

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
                                        steps_per_epoch=iters)

    del train_data_generator, valid_data_generator, test_data_generator
