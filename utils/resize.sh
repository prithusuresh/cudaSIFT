#!/bin/bash

cd ~/cudaSIFT
ffmpeg -i data/vid$1/vid$1_1920.mp4 -s 128x72 data/vid$1/vid$1_128.mp4
ffmpeg -i data/vid$1/vid$1_1920.mp4 -s 384x216 data/vid$1/vid$1_384.mp4
ffmpeg -i data/vid$1/vid$1_1920.mp4 -s 640x360 data/vid$1/vid$1_640.mp4
