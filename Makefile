build_x86:
	docker build --network host \
	-t model_optimizer_x86 -f Dockerfile/dockerfile.x86 .

run:
	docker run -it --network host \
	--gpus all \
	-v ${PWD}:/workspace \
	-v ${HOME}:/srcs \
	model_optimizer_x86 \
	/bin/bash