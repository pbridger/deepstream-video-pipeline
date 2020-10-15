import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

pipeline_name = sys.argv[1]

buffers = 140

Gst.init()
pipeline = Gst.parse_launch(f'''
    nvstreammux name=mux width=384 height=288 batch-size=16 batched-push-timeout=1000000 !
    nvinfer config-file-path=detector.config ! fakesink
    filesrc location=media/in.mp4 num-buffers={buffers} ! decodebin !  mux.sink_0
    filesrc location=media/in.mp4 num-buffers={buffers} ! decodebin !  mux.sink_1
    filesrc location=media/in.mp4 num-buffers={buffers} ! decodebin !  mux.sink_2
    filesrc location=media/in.mp4 num-buffers={buffers} ! decodebin !  mux.sink_3
    filesrc location=media/in.mp4 num-buffers={buffers} ! decodebin !  mux.sink_4
    filesrc location=media/in.mp4 num-buffers={buffers} ! decodebin !  mux.sink_5
    filesrc location=media/in.mp4 num-buffers={buffers} ! decodebin !  mux.sink_6
    filesrc location=media/in.mp4 num-buffers={buffers} ! decodebin !  mux.sink_7
''')

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
    open(f'logs/{pipeline_name}.pipeline.dot', 'w').write(
        Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
    )
    pipeline.set_state(Gst.State.NULL)
