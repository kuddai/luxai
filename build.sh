#!/bin/sh

docker build `dirname $(realpath $0)` -t lux_ai --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USER=$USER