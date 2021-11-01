#!/bin/bash

wget https://www.dropbox.com/s/6shqbfm36qgpn1m/Deeplab_ensemble.pth?dl=1
# path=/root/scripts/AppNexus/tagid/

python3 hw1_2.py  $1 $2

exit 0
