# 构建时增加带时间的 tag（如 model_optimizer:x86-20260202-153045），并保留无时间后缀的 tag 指向本次构建，便于 run_* 直接使用
DOCKER_TS := $(shell date +%Y%m%d-%H%M%S)

build_x86:
	docker build --progress=plain --network host \
	-t model_optimizer:x86-$(DOCKER_TS) \
	-t model_optimizer:x86 \
	-f Dockerfile/dockerfile.x86 .

build_thor:
	docker build --progress=plain --network host \
	-t model_optimizer:thor \
	-t model_optimizer:thor-$(DOCKER_TS) \
	-f Dockerfile/dockerfile.thor .

run_x86:
	docker run -it --network host \
	--gpus all \
	-v ${PWD}:/workspace \
	-v ${HOME}:/srcs \
	-v ${HOME}/.cache:~/.cache \
	--shm-size=4g \
	--env PYTHONPATH=/opt/openpi/lib/python3.12/site-packages/:/workspace/third_party/openpi/src/:/workspace/third_party/openpi/packages/openpi-client/src/ \
	model_optimizer:x86 \
	/bin/bash

run_thor:
	docker run -itd --name model_optimizer_thor --network host \
	--runtime nvidia \
	-v ${PWD}:/workspace \
	-v ${HOME}:/srcs \
	-v ${HOME}/.cache:/srcs/.cache \
	--shm-size=4g \
	--env PYTHONPATH=/opt/openpi/lib/python3.12/site-packages/:/srcs/openpi/src/:/srcs/openpi/packages/openpi-client/src/ \
	model_optimizer:thor \

into_thor:
	docker exec -it model_optimizer_thor /bin/bash