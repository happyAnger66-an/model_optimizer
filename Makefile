build_x86:
	docker build --progress=plain --network host \
	-t model_optimizer:x86 -f Dockerfile/dockerfile.x86 .

build_thor:
	docker build --progress=plain --network host \
	-t model_optimizer:thor -f Dockerfile/dockerfile.thor .

run_x86:
	docker run -it --network host \
	--gpus all \
	-v ${PWD}:/workspace \
	-v ${HOME}:/srcs \
	-v ${HOME}/.cache:/srcs/.cache \
	--shm-size=4g \
	--env PYTHONPATH=/opt/openpi/lib/python3.12/site-packages/:/srcs/sources/opensrc/robot/openpi/src/:/srcs/sources/opensrc/robot/openpi/packages/openpi-client/src/ \
	model_optimizer:x86 \
	/bin/bash

run_thor:
	docker run -it --network host \
	--runtime nvidia \
	-v ${PWD}:/workspace \
	-v ${HOME}:/srcs \
	-v ${HOME}/.cache:/srcs/.cache \
	--shm-size=4g \
	--env PYTHONPATH=/opt/openpi/lib/python3.12/site-packages/:/srcs/openpi/src/:/srcs/openpi/packages/openpi-client/src/ \
	model_optimizer:thor \