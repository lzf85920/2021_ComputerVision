#!/bin/bash

wget https://www.dropbox.com/s/xe97f68p4p3rl3x/vit_v2_ensemble.pth?dl=1

python3 hw3_1.py  $1 $2

exit 0