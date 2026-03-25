import setuptools
from setuptools import find_packages

# Required and optional dependencies ###############################################################
# 核心依赖（CLI、registry 等基础功能）
required_deps = [
    "numpy>=1.24",
    "tqdm>=4.65",
    "addict>=2.4",
    "termcolor>=2.0",
    "colorama>=0.4",
]

optional_deps = {
    # 完整功能: 量化、导出、ONNX、YOLO、WebUI
    "all": [
        "torch>=2.0",
        "torchvision",
        "torchaudio",
        "nvidia-modelopt[all]",
        "transformers>=4.35",
        "safetensors",
        "onnx>=1.12.0,<2.0",
        "onnxruntime-gpu",
        "onnx2pytorch",
        "onnxslim>=0.1.71",
        "ultralytics>=8.0",
        "gradio>=4.0",
        "datasets>=2.14",
        "psutil",
        "pyyaml",
        "pandas",
        "opencv-python-headless",
    ],
    "pi05": [
        "jax[cuda12]",
        "flax>=0.10.0",
        "chex",
        "ml-collections>=1.0.0",
        "tyro>=0.9.5",
        "sentencepiece",
        "datasets>=3.0",
        "av>=15.0.0,<16.0.0",
        "gcsfs",
    ],
    "dev-test": [
        "pytest>=7.0",
    ],
    "dev-lint": [],
    "dev-docs": [],
    "dev-build": [],
}

# create "compound" optional dependencies
optional_deps["all"] = [
    deps for k in optional_deps if not k.startswith("dev") for deps in optional_deps[k]
]
optional_deps["dev"] = [deps for k in optional_deps for deps in optional_deps[k]]

import os
def get_console_scripts() -> list[str]:
    console_scripts = ["model-optimizer-cli = model_optimizer.cli:main"]
    if os.getenv("ENABLE_SHORT_CONSOLE", "1").lower() in ["true", "y", "1"]:
        console_scripts.append("model-opt = model_optimizer.cli:main")

    return console_scripts

def main():
    setuptools.setup(
        name="model_optimizer",
        # version 由 pyproject.toml dynamic + setuptools-scm 写入
        description="Model Optimizer: a unified model optimization and deployment toolkit.",
        long_description="Checkout http://gitlab.anyverse.work/dev/model-optimizer.git for more information.",
        long_description_content_type="text/markdown",
        author="zhangxiaoan",
        url="http://gitlab.anyverse.work/dev/model-optimizer.git",
        license="Apache 2.0",
        license_files=("LICENSE_HEADER",),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        python_requires=">=3.10,<3.13",
#        install_requires=required_deps,
#        extras_require=optional_deps,
        entry_points={"console_scripts": get_console_scripts()},
        packages=find_packages("src"),
        package_dir={"": "src"},
        package_data={"model_optimizer": ["**/*.h", "**/*.cpp", "**/*.cu"]},
    )

if __name__ == "__main__":
    main()