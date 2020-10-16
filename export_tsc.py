import sys
import subprocess
import importlib
import argparse

import torch

def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--batch-dim', type=int, default=8)
    a.add_argument('--height-dim', type=int, default=300)
    a.add_argument('--width-dim', type=int, default=300)
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

    device = torch.device('cuda:0')

    threshold = 0.4
    model_precision = 'fp16'

    image_nchw = (torch.randn((args.batch_dim, 3, args.height_dim, args.width_dim)) * 255).to(device, torch.float16)
    ssd_model = ds_ssd300.SSD300(threshold, model_precision, args.batch_dim)
    tensorrt_model = ds_trt.TensorRTPart(ssd_model).to(device)
    torchscript_model = ds_tsc.TorchScriptPart(ssd_model).to(device)

    # sanity test
    intermediate_result = tensorrt_model(image_nchw)
    intermediate_result = tuple(r.squeeze(0) for r in intermediate_result)
    print(torchscript_model(*intermediate_result))

    with torch.jit.optimized_execution(should_optimize=True):
        for gpu_id in range(2):
            traced_module = torch.jit.trace(
                torchscript_model.to(torch.device('cuda', gpu_id)),
                tuple(r.to(torch.device('cuda', gpu_id)) for r in intermediate_result),
            )
            traced_module.save(f'checkpoints/{args.tsc_module_name}.tsc.pth.{gpu_id}')

#         print('sanity output:')
#         print(traced_module(intermediate_result))
#         print('device independence:')
#         traced_module = traced_module.to(torch.device('cuda:1'))
#         intermediate_result = intermediate_result.to(torch.device('cuda:1'))
#         print(traced_module(intermediate_result))

