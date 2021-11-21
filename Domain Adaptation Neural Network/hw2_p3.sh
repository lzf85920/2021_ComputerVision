#!/bin/bash

wget https://www.dropbox.com/s/dgupd3goj9fkyy3/DANN_SVHN_to_MNIST-M_best.pth?dl=1
wget https://www.dropbox.com/s/izu6w84gu6xphh6/MNIST-M_to_USPS_best.pth?dl=1
wget https://www.dropbox.com/s/7j6tk5r0a031zt6/USPS_to_SVHN_best.pth?dl=1

python3 hw2_p3.py $1 $2 $3

exit 0