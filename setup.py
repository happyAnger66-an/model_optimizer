import setuptools
from setuptools import find_packages
from setuptools_scm import get_version

# TODO: Set fallback_version to X.Y.Z release version when creating the release branch
version = get_version(root=".", fallback_version="0.0.0")

# Required and optional dependencies ###############################################################
required_deps = [
    # Common
#    "numpy",
#    "tqdm",
#    "torch>=2.6",
#    "torchprofile>=0.0.4",
]

optional_deps = {
    "onnx": [
    ],
    "hf": [
    ],
    # linter tools
    "dev-lint": [
    ],
    # testing
    "dev-test": [
    ],
    # docs
    "dev-docs": [
    ],
    # build/packaging tools
    "dev-build": [
    ],
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
        version=version,
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
        install_requires=required_deps,
        extras_require=optional_deps,
        entry_points={"console_scripts": get_console_scripts()},
        packages=find_packages("src"),
        package_dir={"": "src"},
        package_data={"model_optimizer": ["**/*.h", "**/*.cpp", "**/*.cu"]},
    )

if __name__ == "__main__":
    main()