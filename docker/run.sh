#!/bin/bash

gpu=$1
shift

DIR=/project

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU="$gpu" ${cmd} run -it \
        --net host \
        --name henry-$gpu \
        -v `pwd`/:$DIR:rw \
        -t henry \
        $@

# -d -it 
