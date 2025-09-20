#!/usr/bin/env bash

# Added pre-build steps with venv recreation for clean build
pushd /workspace/pytorch2.0_ttnn/ >/dev/null
source /workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/third-party/tt-metal/python_env/bin/activate
pip install -e . --use-pep517 --no-cache-dir --no-build-isolation

popd >/dev/null

pushd /workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/ >/dev/null
./build_cpp_extension.sh
popd >/dev/null

python /workspace/pytorch2.0_ttnn/scripts/test_script.py

# pushd /workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/third-party/tt-metal/build/bin >/dev/null
# ./tt-metal-trace
# popd >/dev/null
