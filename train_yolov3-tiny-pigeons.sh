#!/bin/sh
./darknet/darknet detector train ./darknet/pigeons_cfg/pigeons.data ./darknet/pigeons_cfg/yolov3-tiny-pigeons.cfg ./darknet/darknet53.conv.74 -dont_show
