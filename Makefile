
DOCKER_CMD := docker run -it --rm --gpus=all --privileged=true --ulimit core=0 --ipc=host -v $(shell pwd):/app -v $(shell pwd)/../pbinfer/pbdsinfer:/opt/nvidia/deepstream/deepstream-5.0/sources/libs/nvdsinfer
DOCKER_PY_CMD := ${DOCKER_CMD} --entrypoint=python
DOCKER_NSYS_CMD := ${DOCKER_CMD} --entrypoint=nsys
PROFILE_CMD := profile -t cuda,cublas,cudnn,nvtx,osrt --force-overwrite=true --duration=30

.PHONY: sleep profile_pipeline_%
.PRECIOUS: logs/ds_trt_tsc_%.qdrep

### External - to be used from outside the container ###


build-container: docker/Dockerfile
	docker build -f $< -t deepstream-video-pipeline:latest .


run-container: #build-container
	${DOCKER_CMD} deepstream-video-pipeline:latest


logs/cli.pipeline.dot:
	${DOCKER_CMD} --entrypoint=gst-launch-1.0 deepstream-video-pipeline:latest filesrc location=media/in.mp4 num-buffers=200 ! decodebin ! progressreport update-freq=1 ! fakesink sync=true


logs/%.pipeline.dot: %.py
	${DOCKER_PY_CMD} deepstream-video-pipeline:latest $<


logs/%.qdrep: %.py
	${DOCKER_NSYS_CMD} deepstream-video-pipeline:latest ${PROFILE_CMD} -o $@ python $<


%.pipeline.png: logs/%.pipeline.dot
	dot -Tpng -o$@ $< && rm -f $<


%.output.svg: %.rec
	cat $< | svg-term > $@


%.rec:
	asciinema rec $@ -c "$(MAKE) --no-print-directory logs/$*.pipeline.dot sleep"


### Internal - to be used from within the container ###


checkpoints/ds_tsc_%.tsc.pth: export_tsc.py ds_tsc_%.py ds_trt_%.py ds_ssd300_%.py Makefile
	python $< --ssd-module-name=ds_ssd300_$* --trt-module-name=ds_trt_$* --tsc-module-name=ds_tsc_$* --batch-dim=16


checkpoints/ds_trt_%.engine: export_trt_engine.py ds_trt_%.py ds_ssd300_%.py Makefile
	python $< --ssd-module-name=ds_ssd300_$* --trt-module-name=ds_trt_$* --batch-dim=16


build/Makefile: CMakeLists.txt
	mkdir -p build && \
	cd build && \
	cmake -DCMAKE_INSTALL_PREFIX=/opt/libtorch ..


build/libds_trt_tsc_bridge.so: build/Makefile ds_trt_tsc_bridge.cpp
	cd build && cmake --build . --config Debug


run_pipeline_%: checkpoints/ds_tsc_%.tsc.pth checkpoints/ds_trt_%.engine build/libds_trt_tsc_bridge.so
	DS_TSC_PTH_PATH="$<" \
    gst-launch-1.0 nvstreammux name=mux width=384 height=288 batch-size=16 batched-push-timeout=1000000 ! \
    nvinfer config-file-path=detector.config ! fakesink \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_0 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_1 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_2 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_3 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_4 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_5 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_6 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_7


debug_pipeline_%: checkpoints/ds_tsc_%.tsc.pth checkpoints/ds_trt_%.engine build/libds_trt_tsc_bridge.so
	DS_TSC_PTH_PATH="$<" \
    gdb --args gst-launch-1.0 nvstreammux name=mux width=384 height=288 batch-size=16 batched-push-timeout=1000000 ! \
    nvinfer config-file-path=detector.config ! fakesink \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_0 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_1 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_2 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_3 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_4 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_5 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_6 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_7


logs/ds_trt_tsc_%.qdrep: checkpoints/ds_tsc_%.tsc.pth checkpoints/ds_trt_%.engine build/libds_trt_tsc_bridge.so
	DS_TSC_PTH_PATH="$<" nsys ${PROFILE_CMD} -o $@ \
    gst-launch-1.0 nvstreammux name=mux width=384 height=288 batch-size=16 batched-push-timeout=1000000 ! \
    nvinfer config-file-path=detector.config ! fakesink \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_0 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_1 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_2 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_3 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_4 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_5 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_6 \
    filesrc location=media/in.mp4 num-buffers=140 ! decodebin !  mux.sink_7


profile_pipeline_%: logs/ds_trt_tsc_%.qdrep
	ls -l $<


sleep:
	@sleep 2
	@echo "---"


