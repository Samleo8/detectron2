#!/bin/sh

cd demo

python demo.py --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml \
  --input ../images/test.jpg \
  --opts MODEL.WEIGHTS ../model_final_280758.pkl

cd -