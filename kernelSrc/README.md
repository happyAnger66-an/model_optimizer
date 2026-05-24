# 1. AOT 编译（需 SM100/SM110 GPU + cutlass-dsl 4.4.1）
pip install -r kernelSrc/requirements-cutedsl.txt
model-optimizer-cli kernels build --config config/cutedsl_build.yaml

# 2. 构建 Plugin
mkdir -p cpp/build && cd cpp/build
cmake .. -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR -DENABLE_CUTE_DSL=fmha -DCUTE_DSL_ARTIFACT_TAG=sm_110
make -j$(nproc)

# 3. 导出 ONNX（π0.5 LLM，含 FmhaD256AttentionPlugin）
model-optimizer-cli export --model_name pi05_libero/llm_with_cutedsl ...

# 4. TRT 建引擎
model-optimizer-cli build --build_cfg config/build_configs/llm_cutedsl_build_cfg.py ...