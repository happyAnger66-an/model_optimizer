build_x86:
	docker build --network host \
	-t model_optimizer_x86 -f Dockerfile/dockerfile.x86 .

run:
	docker run -it --network host \
	--gpus all \
	-v ${PWD}:/workspace \
	-v ${HOME}:/srcs \
	--env PYTHONPATH=$PYTHONPATH:/srcs/sources/e2e_model/openpi/src/:/srcs/sources/e2e_model/openpi/packages/openpi-client/src \
	model_optimizer_x86 \
	/bin/bash