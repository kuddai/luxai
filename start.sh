#!/bin/sh
docker run -d -it --gpus all -v /tmp/lux_ai:/tmp/lux_ai -v `pwd`:/lux_ai \
  -p 8888:8888 \
  -p 0.0.0.0:6006:6006 \
  -p 6007:6007 \
  -p 8000:8000 \
  -p 5000:5000 \
  -p 3000:3000 \
  --name lux_ai \
  --ipc=host \
  lux_ai
./attach.sh
