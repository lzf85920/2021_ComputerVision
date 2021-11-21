#!/bin/bash

wget https://www.dropbox.com/s/uq7bk5lh3muxr0g/DANN_SVHN_to_MNIST-M_improved_best.pth?dl=1
wget https://www.dropbox.com/s/3djoihpd79hlhub/MNIST-M_to_USPS_improved_best.pth?dl=1
wget https://www.dropbox.com/s/wdlbi0rdo1xklyn/USPS_to_SVHN_improved_best.pth?dl=1

python3 hw2_bonus.py $1 $2 $3

exit 0