#!/bin/bash

pip3 install -r requirements.txt

python3 knn.py

python3 make_gif.py

echo "finished!"