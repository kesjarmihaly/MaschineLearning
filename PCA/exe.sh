#!/bin/bash

pip3 install requirements.txt

python3 pca_sample.py

python3 model.py

python3 pca.py

python3 eval.py

python3 make_graph.py

echo "Finish!"

