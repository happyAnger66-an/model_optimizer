build_x86:
	docker build --network host \
	-t model_optimizer_x86 -f Dockerfile/dockerfile.x86 .

run:
	docker run -it --network host \
	--gpus all \
	-v ${PWD}:/workspace \
	-v ${HOME}:/srcs \
	--shm-size=4g \
	--env PYTHONPATH=${PYTHONPATH}:PYTHONPATH=:/srcs/sources/opensrc/robot/openpi/src/:/srcs/sources/opensrc/robot/openpi/packages/openpi-client/src/ \
	model_optimizer_x86:0122 \
	/bin/bash