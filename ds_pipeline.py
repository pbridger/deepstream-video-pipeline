import sys
import argparse
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

p = argparse.ArgumentParser()
p.add_argument('--name', default='default')
p.add_argument('--buffers', default=512)
p.add_argument('--decodebin', default=False, action='store_true')
p.add_argument('--batch-size', default=8, type=int)
p.add_argument('--gpus', default=1, type=int)

args = p.parse_args()

pipeline_cmd = ''
for gpu_id in range(args.gpus):
    pipeline_cmd += f'''\
nvstreammux name=mux{gpu_id} gpu-id={gpu_id} enable-padding=1 width=300 height=300 batch-size={args.batch_size} batched-push-timeout=1000000 !
nvinfer config-file-path=detector.config gpu-id={gpu_id} batch-size={args.batch_size} ! fakesink
'''
    for filesrc_id in range(args.batch_size // 2):
        if args.decodebin:
            pipeline_cmd += f'filesrc location=media/in.mp4 num-buffers={args.buffers} ! decodebin ! mux{gpu_id}.sink_{filesrc_id} \n'
        else:
            pipeline_cmd += f'filesrc location=media/in.mp4 num-buffers={args.buffers} ! qtdemux ! h264parse ! nvv4l2decoder gpu-id={gpu_id} ! mux{gpu_id}.sink_{filesrc_id} \n'

print(pipeline_cmd)

Gst.init()
pipeline = Gst.parse_launch(pipeline_cmd)

pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        msg = pipeline.get_bus().timed_pop_filtered(
            Gst.SECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        if msg:
            text = msg.get_structure().to_string() if msg.get_structure() else ''
            msg_type = Gst.message_type_get_name(msg.type)
            print(f'{msg.src.name}: [{msg_type}] {text}')
            break
finally:
    open(f'logs/{args.name}_{args.gpus}gpu_batch{args.batch_size}.pipeline.dot', 'w').write(
        Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
    )
    pipeline.set_state(Gst.State.NULL)
