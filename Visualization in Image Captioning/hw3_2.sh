#!/bin/bash

python3 ./hw3_2_download.py
python3 ./catr/predict.py --path $1 --output $2

exit 0