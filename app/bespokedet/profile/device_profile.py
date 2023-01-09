# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import sys
import os
from timeit import default_timer as timer
import argparse

import numpy as np
import tensorflow as tf
import json
import onnxruntime as rt

from onnx import ModelProto
import tensorrt as trt
import inference as inf
import engine as eng
import tqdm


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def run():

    parser = argparse.ArgumentParser(description='Device Profiler', add_help=False)
    parser.add_argument('--mode', type=str, default="onnx_cpu", help='model')
    parser.add_argument('--prefix', type=str, default="", help='model')
    args = parser.parse_args()
    mode = args.mode

    batch_size = 1
    num_rounds = 100
    onnxpath = "onnx_models"

    with open("meta.json", "r") as f:
        meta = json.load(f)

    data = {
        "data": {},
    }

    onlyfiles = [f for f in os.listdir(onnxpath) if os.path.isfile(os.path.join(onnxpath, f))]
    for filename in tqdm.tqdm(onlyfiles):
        filepath = os.path.join(onnxpath, filename)
        filename_base = os.path.splitext(filename)[0]

        model = ModelProto()
        with open(filepath, "rb") as f:
            model.ParseFromString(f.read())

            d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
            d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
            d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value

            o0 = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
            o1 = model.graph.output[0].type.tensor_type.shape.dim[2].dim_value
            o2 = model.graph.output[0].type.tensor_type.shape.dim[3].dim_value
            input_name = model.graph.input[0].name  # for onnx
            output_name = model.graph.output[0].name # for onnx
            shape = [batch_size , d0, d1 ,d2]

            if mode == "trt":
                engine_name = filename_base + ".plan"
                engine = eng.build_engine(filepath, shape= shape)
                eng.save_engine(engine, engine_name)
                engine = eng.load_engine(trt_runtime, engine_name)
                h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, batch_size, trt.float32)

            else:

                if mode == "onnx_cpu":
                    providers = ['CPUExecutionProvider']
                    DEVICE_NAME = "cpu"
                    DEVICE_INDEX = 0
                elif mode == "onnx_gpu":
                    providers = [('CUDAExecutionProvider', {"device_id":0})]
                    DEVICE_NAME = "cuda"
                    DEVICE_INDEX = 0
                m = rt.InferenceSession(filepath, providers=providers)

                input_data = np.array(np.random.rand(*shape), dtype=np.float32)
                x_ortvalue = rt.OrtValue.ortvalue_from_numpy(input_data, DEVICE_NAME, DEVICE_INDEX)
                io_binding = m.io_binding()
                io_binding.bind_input(
                    name=input_name,
                    device_type=x_ortvalue.device_name(),
                    device_id=DEVICE_INDEX,
                    element_type=input_data.dtype,
                    shape=x_ortvalue.shape(),
                    buffer_ptr=x_ortvalue.data_ptr())
                io_binding.bind_output(output_name)

                # warming up
                for i in range(10):
                    onnx_pred = m.run_with_iobinding(io_binding)

            total_t = 0
            for _ in range(num_rounds):
                
                im = np.array(np.random.rand(batch_size, d2, d0, d1), dtype=np.float32)
                start = timer()
                if mode == "trt":
                    inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, batch_size, o0, o1)
                else:
                    m.run_with_iobinding(io_binding)
                time_ = float(timer() - start)
                total_t = total_t + time_
            avg_time = (total_t / float(num_rounds))
            data["data"][filename_base] = {args.prefix+args.mode:avg_time}

    with open("result.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    run()
