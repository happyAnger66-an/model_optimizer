# 1. AOT 编译（需 SM100/SM110 GPU + cutlass-dsl 4.4.1）
pip install -r kernelSrc/requirements-cutedsl.txt

```bash
python kernelSrc/build_cutedsl.py --kernels fmha --gpu_arch sm_110 -j 4 --output_dir cpp/kernels/cuteDSLArtifact
```

# 2. 构建 Plugin
```bash
mkdir -p cpp/build && cd cpp/build
cmake -S cpp -B cpp/build/ -DCMAKE_BUILD_TYPE=Release   -DTRT_PACKAGE_DIR=/usr/include/aarch64-linux-gnu/   -DENABLE_CUTE_DSL=fmha   -DCUTE_DSL_ARTIFACT_TAG=sm_110
cmake --build cpp/build/ -j 10
```

# 3. 导出 ONNX（π0.5 LLM，含 FmhaD256AttentionPlugin）
```bash
model-optimizer-cli export --model_name pi05_libero/llm_with_cutedsl ...
```

# 4. TRT 建引擎
```bash
model-optimizer-cli build --build_cfg config/build_configs/llm_cutedsl_build_cfg.py ...
```