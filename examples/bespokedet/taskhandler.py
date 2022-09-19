from train import *
from automl.efficientdet.tf2 import efficientdet_keras
from automl.efficientdet.tf2 import train_lib
from automl.efficientdet.tf2 import util_keras
from automl.efficientdet import hparams_config
from automl.efficientdet import utils

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
    train_dataset = dataloader.InputReader(
        train_file_pattern,
        is_training=True,
        use_fake_data=use_fake_data,
        max_instances_per_image=max_instances_per_image,
        debug=debug)(
            config)

    val_dataset = dataloader.InputReader(
        val_file_pattern,
        is_training=False,
        use_fake_data=use_fake_data,
        max_instances_per_image=max_instances_per_image,
        debug=debug)(
            config)

    return train_dataset, val_dataset


def load_tl_dataset(config):
    dataset = load_dataset_(config)
    def inject_dummy(images, labels):
        dummy = tf.ones((images.shape[0],))
        return images, dummy
    dataset = dataset.map(inject_dummy)
    return dataset

def extract_backbone(config):
    config_ = to_hparam_config(config)
    detmodel = train_lib.EfficientDetNetTrain(config=config_)
    detmodel.build((config_.batch_size, *config_.image_size, 3))
    util_keras.restore_ckpt(
        detmodel, config["base_ckpt_path"], config_.moving_average_decay, skip_mismatch=True)

    detmodel = setup_model_(config, detmodel)
    return detmodel

def replace_backbone(detmodel, new_backbone):
    detmodel.backbone = new_backbone
 
def post_prep_(config, model):
    config_ = to_hparam_config(config)
    detmodel = train_lib.EfficientDetNetTrain(config=config_)
    detmodel.backbone = FeatureModel(model)
    detmodel.build((config_.batch_size, *config_.image_size, 3))
    detmodel = setup_model_(config, detmodel)
    return detmodel

def validate_(config, model):
    train_dataset, valid_dataset = load_dataset_(config)
    coco_eval = train_lib.COCOCallback(valid_dataset, 1)
    coco_eval.set_model(model)
    eval_results = coco_eval.on_epoch_end(0)
    print(eval_results)

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

    num_iters = config["examples_per_epoch"] // config["batch_size"]
    num_iters_val = config["eval_samples"] // config["batch_size"]
    config["profile"] = False
    callbacks = train_lib.get_callbacks(params=config)

    train(
        model,
        train_dataset,
        val_dataset,
        num_iters,
        num_iters_val,
        epochs=config["epochs"],
        prefix=prefix,
        save_dir=save_dir,
        callbacks=callbacks,
        use_hvd=config["use_hvd"])
