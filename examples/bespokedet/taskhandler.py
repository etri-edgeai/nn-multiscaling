import copy
import tqdm

from train import *
from automl.efficientdet.tf2 import efficientdet_keras
from automl.efficientdet.tf2 import train_lib
from automl.efficientdet.tf2 import util_keras
from automl.efficientdet import hparams_config
from automl.efficientdet import utils
import horovod.tensorflow.keras as hvd

import dataloader

class FeatureModel(tf.keras.Model):

    def __init__(self, model):
        super(FeatureModel, self).__init__()
        self.model = model

    def call(self, inputs, training=False, features_only=False):
        ret = self.model(inputs, training)
        if features_only:
            return [ret[-1]] + ret[1:]
        else:
            return ret            

def build_config(partial_config):
    """ Converting a dict-based configuration to a hparam configuration

        Args.
            
            config: dict, a configuration dict

        Returns.

            Config object  

    """
    config_ = to_hparam_config(partial_config)
    return config_.as_dict()


def dump_config(config):
    config = copy.deepcopy(config)
    config["image_size"] = config["image_size"][0]
    return config

def to_hparam_config(config):
    """ Converting a dict-based configuration to a hparam configuration

        Args.
            
            config: dict, a configuration dict

        Returns.

            Config object  

    """
    config_ = hparams_config.get_efficientdet_config(config["model_name"])
    config_.override(config, True)
    config_.image_size = utils.parse_image_size(config_.image_size)
    return config_

def load_dataset_(config):
    """Dataset Loader from a configuration

        Args.

            config: dict, a configuration dict

        Returns.
            
            Train dataset, Validation dataset

    """
    train_file_pattern = config["train_file_pattern"]
    val_file_pattern = config["val_file_pattern"]
    use_fake_data = config["use_fake_data"]
    max_instances_per_image = config["max_instances_per_image"]
    debug = config["debug"]

    train_dataset = dataloader.InputReader(
        train_file_pattern,
        is_training=True,
        use_fake_data=use_fake_data,
        max_instances_per_image=max_instances_per_image,
        debug=debug)(
            copy.deepcopy(config))

    val_dataset = dataloader.InputReader(
        val_file_pattern,
        is_training=False,
        use_fake_data=use_fake_data,
        max_instances_per_image=max_instances_per_image,
        debug=debug)(
            copy.deepcopy(config))

    return train_dataset, val_dataset


def load_tl_dataset(config):
    dataset = load_dataset_(config)
    def inject_dummy(images, labels):
        dummy = tf.ones((images.shape[0],))
        return images, dummy
    train_dataset = dataset[0].map(inject_dummy)
    val_dataset = dataset[1].map(inject_dummy)
    return train_dataset, val_dataset

def extract_backbone(config, ckpt=None):
    config_ = to_hparam_config(config)
    detmodel = train_lib.EfficientDetNetTrain(config=config_)
    detmodel = setup_model_(config, detmodel)
    if ckpt is None:
        util_keras.restore_ckpt(
            detmodel, config["base_ckpt_path"], config_.moving_average_decay, exclude_layers=['class_net', 'optimizer', 'box_net'])
    else:
        util_keras.restore_ckpt(
            detmodel, ckpt, config_.moving_average_decay, exclude_layers=['class_net', 'optimizer', 'box_net'])
    return detmodel.backbone

def replace_backbone(detmodel, new_backbone):
    detmodel.backbone = new_backbone
 
def post_prep_(config, model, pretrained=None, detmodel=None, with_head=False):
    config_ = to_hparam_config(config)
    if detmodel is None:
        detmodel = train_lib.EfficientDetNetTrain(config=config_)
    detmodel.build((config_.batch_size, *config_.image_size, 3))
    if pretrained is not None:
        if with_head:
            util_keras.restore_ckpt(
                detmodel, pretrained, config_.moving_average_decay, exclude_layers=['optimizer'])
        else:
            util_keras.restore_ckpt(
                detmodel, pretrained, config_.moving_average_decay, exclude_layers=['class_net', 'optimizer', 'box_net'])
    else:
        util_keras.restore_ckpt(
            detmodel, config["base_ckpt_path"], config_.moving_average_decay, exclude_layers=['class_net', 'optimizer', 'box_net'], skip_mismatch=True)
    detmodel.backbone = FeatureModel(model)
    detmodel = setup_model_(config, detmodel)
    return detmodel

def validate_(config, model):
    train_dataset, valid_dataset = load_dataset_(config)
    coco_eval = train_lib.COCOCallback(valid_dataset, 1)
    coco_eval.set_model(model)
    eval_results = coco_eval.on_epoch_end(0)
    return eval_results["AP"]

def setup_model_(config, model):
    opt = setup_optimizer(config)

    setup_model(
        model,
        opt,
        image_size=config["image_size"],
        delta=config["delta"],
        iou_loss_type=config["iou_loss_type"],
        min_level=config["min_level"],
        max_level=config["max_level"],
        aspect_ratios=config["aspect_ratios"],
        anchor_scale=config["anchor_scale"],
        alpha=config["alpha"],
        gamma=config["gamma"],
        num_scales=config["num_scales"],
        use_distillation=config["use_distillation"],
        label_smoothing=config["label_smoothing"])

    return model
    
def train_(config, prefix, save_dir, callbacks, model, is_tl=False):

    if is_tl:
        train_dataset, val_dataset = load_tl_dataset(config)
    else:
        train_dataset, val_dataset = load_dataset_(config)

    num_iters = config["num_examples_per_epoch"] // config["batch_size"]
    num_iters_val = config["eval_samples"] // config["batch_size"]
    config["profile"] = False
    if is_tl:
        callbacks = train_lib.get_callbacks(copy.deepcopy(config))
        from butils.loss import BespokeTaskLoss
        bloss = BespokeTaskLoss()
        bloss.mute = True
        opt = setup_optimizer(config)

        loss = {model.output[0].name.split("/")[0]:bloss}
        model.compile(optimizer=opt, loss=loss)

        num_iters = num_iters // hvd.size()
        num_iters_val = num_iters_val // hvd.size()
        model.fit(train_dataset,
                  validation_data=val_dataset,
                  verbose=1 if hvd.local_rank() == 0 else 0,
                  epochs=config["num_epochs"],
                  steps_per_epoch=num_iters,
                  validation_steps=num_iters_val)

    else:
        callbacks = train_lib.get_callbacks(copy.deepcopy(config), val_dataset)
        train(
            model,
            train_dataset,
            val_dataset,
            num_iters,
            num_iters_val,
            epochs=config["num_epochs"],
            prefix=prefix,
            save_dir=save_dir,
            callbacks=callbacks,
            use_hvd=config["use_hvd"])


def backbone_transfer(config, model, teacher):
    opt = setup_optimizer(config)
    train_dataset, val_dataset = load_tl_dataset(config)

    teacher.trainable = False
    num_iters = config["num_examples_per_epoch"] // config["batch_size"]
    num_iters_val = config["eval_samples"] // config["batch_size"]

    first_batch = False
    with tqdm.tqdm(total=num_iters * config["num_epochs"], ncols=120) as pbar:
        for e in range(config["num_epochs"]): 
            sum_loss = 0
            idx = 0
            cnt = 0
            for X, y in train_dataset:
                idx += 1
                cnt += 1
                teacher_logits = teacher(X)
                with tf.GradientTape() as tape:
                    logits = model(X)

                    loss = None
                    for tout, sout in zip(teacher_logits, logits):
                        if len(tout.shape) != len(sout.shape):
                            continue
                        t = tf.cast(tout, tf.float32)
                        s = tf.cast(sout, tf.float32)
                        
                        if loss is None:
                            loss = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(t, s))
                        else:
                            loss += tf.math.reduce_mean(tf.keras.losses.mean_squared_error(t, s))

                    sum_loss += loss

                    #tape = hvd.DistributedGradientTape(tape)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    opt.apply_gradients(zip(gradients, model.trainable_variables))

                    pbar.update(1)

                    if idx % 1000 == 0:
                        print("AVG LOSS:", sum_loss / idx)
                        sum_loss = 0
                        idx = 0
                
                if cnt >= num_iters:
                    break

            if hasattr(train_dataset, "on_epoch_end"):
                train_dataset.on_epoch_end() 
    tf.keras.models.save_model(model, "backbone_trained.h5")
