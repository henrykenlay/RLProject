#!/bin/bash

gpu=$1
shift

DIR=/project

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

dt=$(date '+%d-%m-%Y-%H-%M-%S');

NV_GPU="$gpu" ${cmd} run -d \
        --net host \
        --name henryy-$gpu-$dt \
        -v `pwd`/:$DIR:rw \
        -t henry \
        $@

# -d -it 
