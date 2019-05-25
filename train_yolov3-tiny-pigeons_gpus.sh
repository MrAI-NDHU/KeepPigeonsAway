#!/bin/sh
while true; do
  ./darknet/darknet detector train ./darknet/pigeons_cfg/pigeons.data ./darknet/pigeons_cfg/yolov3-tiny-pigeons.cfg ./darknet/pigeons_weights/yolov3-tiny-pigeons_last.weights -gpus 0,1,2 -dont_show
  echo `TZ="Asia/Taipei" date +"%Y/%m/%d %T"` killed >> darknet_killed_log.txt
  sleep 2
done
