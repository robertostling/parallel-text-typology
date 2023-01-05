#!/bin/bash

echo "Downloading ASJP data"
wget https://cdstar.shh.mpg.de/bitstreams/EAEA0-E32A-2C2D-B777-0/asjp_dataset.tab.zip
mkdir -p data/resources/asjp
unzip -d data/resources/asjp asjp_dataset.tab.zip

