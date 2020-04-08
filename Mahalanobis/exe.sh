#!/bin/bash

pip3 install -r requirements.txt

python3 cov.py

python3 iris.py

python3 mahalanobis.py

echo "finished!"