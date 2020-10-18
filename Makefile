
DOCKER_CMD := docker run -it --rm --gpus=all --privileged=true --ulimit core=0 --ipc=host -v $(shell pwd):/app -v $(shell pwd)/../pbinfer/pbdsinfer:/opt/nvidia/deepstream/deepstream-5.0/sources/libs/nvdsinfer
DOCKER_PY_CMD := ${DOCKER_CMD} --entrypoint=python
DOCKER_NSYS_CMD := ${DOCKER_CMD} --entrypoint=nsys
PROFILE_CMD := profile -t cuda,cublas,cudnn,nvtx,osrt --force-overwrite=true --duration=30 --delay=7

NUM_BUFFERS=512
PARSE_FUNC_NAME = DsTrtTscBridgeDevice

.PHONY: sleep debug_pipeline_%_1gpu_host debug_pipeline_%_1gpu_device 
.PRECIOUS: checkpoints/ds_tsc_%.tsc.pth.0 checkpoints/ds_tsc_%.tsc.pth.1 checkpoints/ds_trt_%.engine

### External - to be used from outside the container ###


build-container: docker/Dockerfile
	docker build --no-cache -f $< -t deepstream-video-pipeline:latest .


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


checkpoints/ds_tsc_%.tsc.pth.0 checkpoints/ds_tsc_%.tsc.pth.1: export_tsc.py ds_tsc_%.py ds_trt_%.py ds_ssd300_%.py
	python $< --ssd-module-name=ds_ssd300_$* --trt-module-name=ds_trt_$* --tsc-module-name=ds_tsc_$* --batch-dim=16


checkpoints/ds_trt_%.engine: export_trt_engine.py ds_trt_%.py ds_ssd300_%.py
	python $< --ssd-module-name=ds_ssd300_$* --trt-module-name=ds_trt_$* --batch-dim=16 --output-names=$(shell python ds_trt_$*.py)


build/Makefile: CMakeLists.txt
	mkdir -p build && \
	cd build && \
	cmake -DCMAKE_INSTALL_PREFIX=/opt/libtorch ..


build/libds_trt_tsc_bridge.so: build/Makefile ds_trt_tsc_bridge.cpp
	cd build && cmake --build . --config Debug


detector_%.config: detector.config
	cat $< | sed 's/{engine_name}/$*/g' | sed 's/{parse_func_name}/${PARSE_FUNC_NAME}/g' >$@



logs/%_batch16.pipeline.dot: ds_pipeline.py checkpoints/ds_trt_%.engine checkpoints/ds_tsc_%.tsc.pth.0 checkpoints/ds_tsc_%.tsc.pth.1 build/libds_trt_tsc_bridge.so detector_%.config
	cat detector_$*.config
	DS_TSC_INPUTS="$(shell python ds_trt_$*.py)" DS_TSC_PTH_PATH="checkpoints/ds_tsc_$*.tsc.pth." python $< --batch-size=16 --name=$* --buffers=${NUM_BUFFERS} --gpus=${GPUS}
	rm detector_$*.config


debug_pipeline_%: ds_pipeline.py checkpoints/ds_trt_%.engine checkpoints/ds_tsc_%.tsc.pth.0 checkpoints/ds_tsc_%.tsc.pth.1 build/libds_trt_tsc_bridge.so detector_%.config
	DS_TSC_INPUTS="$(shell python ds_trt_$*.py)" DS_TSC_PTH_PATH="checkpoints/ds_tsc_$*.tsc.pth." gdb --args python $< --batch-size=16 --name=$* --buffers=80 --gpus=1
	rm detector_$*.config


logs/ds_%.qdrep: ds_pipeline.py checkpoints/ds_trt_%.engine checkpoints/ds_tsc_%.tsc.pth.0 checkpoints/ds_tsc_%.tsc.pth.1 build/libds_trt_tsc_bridge.so detector_%.config detector_%.config
	cat detector_$*.config
	DS_TSC_INPUTS="$(shell python ds_trt_$*.py)" DS_TSC_PTH_PATH="checkpoints/ds_tsc_$*.tsc.pth." nsys ${PROFILE_CMD} -o $@ python $< --batch-size=16 --name=$* --buffers=${NUM_BUFFERS} --gpus=${GPUS}
	rm detector_$*.config


run_pipeline_%_1gpu_host: GPUS=1
run_pipeline_%_1gpu_host: PARSE_FUNC_NAME=DsTrtTscBridgeHost
run_pipeline_%_1gpu_host: NUM_BUFFERS=256
run_pipeline_%_1gpu_host: logs/%_batch16.pipeline.dot
	mv $< logs/ds_$*_1gpu_batch16_host.pipeline.dot

run_pipeline_%_2gpu_host: GPUS=2
run_pipeline_%_2gpu_host: PARSE_FUNC_NAME=DsTrtTscBridgeHost
run_pipeline_%_2gpu_host: NUM_BUFFERS=256
run_pipeline_%_2gpu_host: logs/%_batch16.pipeline.dot
	mv $< logs/ds_$*_2gpu_batch16_host.pipeline.dot

run_pipeline_%_1gpu_device: GPUS=1
run_pipeline_%_1gpu_device: PARSE_FUNC_NAME=DsTrtTscBridgeDevice
run_pipeline_%_1gpu_device: logs/%_batch16.pipeline.dot
	mv $< logs/ds_$*_1gpu_batch16_device.pipeline.dot

run_pipeline_%_2gpu_device: GPUS=2
run_pipeline_%_2gpu_device: PARSE_FUNC_NAME=DsTrtTscBridgeDevice
run_pipeline_%_2gpu_device: logs/%_batch16.pipeline.dot
	mv $< logs/ds_$*_2gpu_batch16_device.pipeline.dot


debug_pipeline_%_1gpu_host: PARSE_FUNC_NAME=DsTrtTscBridgeHost
debug_pipeline_%_1gpu_host: debug_pipeline_%

debug_pipeline_%_1gpu_device: PARSE_FUNC_NAME=DsTrtTscBridgeDevice
debug_pipeline_%_1gpu_device: debug_pipeline_%


profile_pipeline_%_1gpu_host: GPUS=1
profile_pipeline_%_1gpu_host: PARSE_FUNC_NAME=DsTrtTscBridgeHost
profile_pipeline_%_1gpu_host: NUM_BUFFERS=256
profile_pipeline_%_1gpu_host: logs/ds_%.qdrep
	mv $< logs/ds_$*_1gpu_batch16_host.qdrep

profile_pipeline_%_2gpu_host: GPUS=2
profile_pipeline_%_2gpu_host: PARSE_FUNC_NAME=DsTrtTscBridgeHost
profile_pipeline_%_2gpu_host: NUM_BUFFERS=256
profile_pipeline_%_2gpu_host: logs/ds_%.qdrep
	mv $< logs/ds_$*_2gpu_batch16_host.qdrep

profile_pipeline_%_1gpu_device: GPUS=1
profile_pipeline_%_1gpu_device: PARSE_FUNC_NAME=DsTrtTscBridgeDevice
profile_pipeline_%_1gpu_device: logs/ds_%.qdrep
	mv $< logs/ds_$*_1gpu_batch16_device.qdrep

profile_pipeline_%_2gpu_device: GPUS=2
profile_pipeline_%_2gpu_device: PARSE_FUNC_NAME=DsTrtTscBridgeDevice
profile_pipeline_%_2gpu_device: logs/ds_%.qdrep
	mv $< logs/ds_$*_2gpu_batch16_device.qdrep


sleep:
	@sleep 2
	@echo "---"


