#!/usr/bin/env bash

script_dir=$(cd $(dirname $0);pwd)

echo $(dirname $script_dir)

python $(dirname $script_dir)/serving/main.py