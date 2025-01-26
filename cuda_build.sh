#!/bin/sh

nvcc -o $1 main.cpp world.cu world_priv.cpp world_common.cpp timeline.cpp -I. --compiler-options '-Wall -Wextra -O2'
