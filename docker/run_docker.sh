#!/bin/bash

docker run \
  -it \
  --rm \
  --name dl2 \
  --gpus all \
  --ipc=host \
  -u "$(id -u)":"$(id -g)" \
  -v "$DL2_HOME":/dl2 \
  dl2/base "$@"
