import os
import json
import shutil
import time
import argparse

import yaml
import numpy as np
import tensorflow as tf
import numpy as np

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB2

from bespoke.profile import measure
from bespoke.base.engine import module_load

parser = argparse.ArgumentParser(description='Bespoke profile', add_help=False)
parser.add_argument('--config', type=str, required=True) # dataset-sensitive configuration
parser.add_argument('--model_path', type=str, required=True, help='model')

args = parser.parse_args()

with open(args.config, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(1)

task_class = module_load(config["taskbuilder"])
taskbuilder = task_class(config)

model = taskbuilder.load_model(args.model_path)
model = taskbuilder.prep(model, for_benchmark=True)

print(measure(model, taskbuilder, mode="onnx_cpu"))
print(model.count_params())
