#!/usr/bin/env bash

# I will pass as a parameter tt-metal's tag to chechout like v0.58.0-rc25
METAL_TAG=$1

pushd torch_ttnn/cpp_extension/third-party/tt-metal
git checkout $METAL_TAG
git pull
git submodule sync --recursive
git submodule update --init --recursive --progress
popd
