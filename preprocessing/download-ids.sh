#!/bin/bash

echo "Downloading IDS data"
wget https://cdstar.shh.mpg.de/bitstreams/EAEA0-5F01-8AAF-CDED-0/ids_dataset.cldf.zip
mkdir -p data/resources/ids
unzip ids_dataset.cldf.zip -d data/resources/ids

