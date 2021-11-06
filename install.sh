#!/bin/sh

sudo docker build -t colorgan .
ln -sf /work/marcb/ColorGAN.bkp/experiments/release/ pretrained
