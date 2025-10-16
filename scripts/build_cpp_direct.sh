#!/usr/bin/env bash

export TT_METAL_HOME=$(realpath torch_ttnn/cpp_extension/third-party/tt-metal)
echo "> TT_METAL_HOME: $TT_METAL_HOME"
source $TT_METAL_HOME/python_env/bin/activate

pushd torch_ttnn/cpp_extension/
./build_cpp_extension.sh
popd

# Ensure modern build tooling in the active venv for editable installs
python -m pip install --upgrade pip setuptools wheel setuptools_scm build

pip3 install -e . --use-pep517 --no-build-isolation

python scripts/test_script.py
