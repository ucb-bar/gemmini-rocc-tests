#!/bin/bash

bash clean.sh
python gemmini_data_collection.py
cd ..


if [ "$1" = "tile" ]; then
	sed -i '/-DPRINT_TILE=/c\\t-DPRINT_TILE=1 \\' bareMetalC/Makefile
	echo "Set DPRINT_TILE=1"
else
	sed -i '/-DPRINT_TILE=/c\\t-DPRINT_TILE=0 \\' bareMetalC/Makefile
	echo "Set DPRINT_TILE=0"
fi


./build.sh bareMetalC
cd ../..

if [ "$1" = "tile" ]; then
	echo "Running Spike"
	bash data-collection-spike.sh
else
	echo "Running VCS"
	bash data-collection-vcs.sh
fi	

