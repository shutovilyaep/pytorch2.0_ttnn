#!/usr/bin/env bash

export TT_METAL_HOME=$(realpath torch_ttnn/cpp_extension/third-party/tt-metal)
echo "> TT_METAL_HOME: $TT_METAL_HOME"
source $TT_METAL_HOME/python_env/bin/activate

# Ensure modern build tooling and required deps (torch/torchvision) in this venv BEFORE building the C++ extension
python -m pip install --upgrade pip setuptools wheel setuptools_scm build
python -m pip install 'torch==2.2.1+cpu' 'torchvision==0.17.1+cpu' --extra-index-url https://download.pytorch.org/whl/cpu

pushd torch_ttnn/cpp_extension/
./build_cpp_extension.sh
popd

pip3 install -e . --use-pep517 --no-build-isolation --no-deps

python scripts/test_script.py
