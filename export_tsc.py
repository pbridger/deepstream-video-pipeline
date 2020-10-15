import sys
import subprocess
import importlib
import argparse

import torch

def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--batch-dim', type=int, default=8)
    a.add_argument('--height-dim', type=int, default=288)
    a.add_argument('--width-dim', type=int, default=384)
    a.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a.add_argument('--ssd-module-name', type=str)
    a.add_argument('--trt-module-name', type=str)
    a.add_argument('--tsc-module-name', type=str)
    # a.add_argument('--output-names', default=['image_nchw_out'])
    return a.parse_args()


if __name__ =='__main__':
    args = parse_args()

    ds_ssd300 = importlib.import_module(args.ssd_module_name)
    ds_trt = importlib.import_module(args.trt_module_name)
    ds_tsc = importlib.import_module(args.tsc_module_name)

    dest_path = f'checkpoints/{args.tsc_module_name}.tsc.pth'

    device = torch.device(args.device)

    threshold = 0.4
    model_precision = 'fp16'

    image_nchw = (torch.randn((args.batch_dim, 3, args.height_dim, args.width_dim)) * 255).to(device, torch.float32)
    ssd_model = ds_ssd300.SSD300(threshold, model_precision, device)
    tensorrt_model = ds_trt.TensorRTPart(ssd_model)
    torchscript_model = ds_tsc.TorchScriptPart(ssd_model)

    # sanity test
    intermediate_result = tensorrt_model(image_nchw)
    intermediate_result = intermediate_result[0].squeeze(0) # likely have to modify this when tensorrt_model returns two tensors
    torchscript_model(intermediate_result)

    with torch.jit.optimized_execution(should_optimize=True):
        torch.jit.trace(
            torchscript_model,
            intermediate_result,
        ).save(dest_path)

