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
    a.add_argument('--output-names', default='out')
    return a.parse_args()


if __name__ =='__main__':
    args = parse_args()
    args.output_names = args.output_names.split(',')

    ds_ssd300 = importlib.import_module(args.ssd_module_name)
    ds_trt = importlib.import_module(args.trt_module_name)

    dest_path = f'checkpoints/{args.trt_module_name}.engine'
    onnx_path = f'checkpoints/{args.trt_module_name}.onnx'

    device = torch.device(args.device)

    threshold = 0.4
    model_precision = 'fp16'

    image_nchw = (torch.randn((args.batch_dim, 3, args.height_dim, args.width_dim)) * 255).to(device, torch.float32)
    tensorrt_model = ds_trt.TensorRTPart(ds_ssd300.SSD300(threshold, model_precision, args.batch_dim)).to(device)

    # sanity test
    tensorrt_model(image_nchw)

    torch.onnx.export(
        tensorrt_model,
        image_nchw,
        onnx_path,
        input_names=['image_nchw'],
        output_names=args.output_names,
        dynamic_axes={'image_nchw': {0: 'batch_dim'}, **{o: {1: 'batch_dim'} for o in args.output_names}},
        opset_version=11
    )

    trt_output = subprocess.run([
            'trtexec',
            f'--onnx={onnx_path}',
            f'--saveEngine={dest_path}',
            '--fp16',
            '--explicitBatch',
            f'--minShapes=image_nchw:1x3x{args.height_dim}x{args.width_dim}',
            f'--optShapes=image_nchw:{args.batch_dim}x3x{args.height_dim}x{args.width_dim}',
            f'--maxShapes=image_nchw:{args.batch_dim}x3x{args.height_dim}x{args.width_dim}',
            '--buildOnly'
        ],
        shell=False,
        check=True,
        capture_output=True,
        text=True
    )
            # f'--shapes=image_nchw:1x3x{args.height_dim}x{args.width_dim}',

    print(trt_output.args)
    print(trt_output.stdout)
    print(trt_output.stderr)


