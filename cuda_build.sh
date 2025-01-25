#!/bin/sh

nvcc -g -G -o $1 main.cpp world.cu world_priv.c world_common.c timeline.c -I. --compiler-options '-Wall -Wextra -O2' 
