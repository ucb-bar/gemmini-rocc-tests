#!/bin/bash

bash clean.sh
python gemmini_data_collection.py
cd ..


if [ "$1" = "tile" ]; then
	sed -i '/-DPRINT_TILE=/c\\t-DPRINT_TILE=1 \\' bareMetalC/Makefile
	echo "Set DPRINT_TILE=1"
elif [ "$1" = "cycle" ]; then
	sed -i '/-DPRINT_TILE=/c\\t-DPRINT_TILE=0 \\' bareMetalC/Makefile
	echo "Set DPRINT_TILE=0"
else
	echo "Invalid first parameter passed into gen-data.sh: should be 'tile' or 'cycle'"
	exit 1
fi


./build.sh bareMetalC
cd ../..

if [ "$1" = "tile" ]; then
	echo "Running Spike"
	bash data-collection-spike.sh
elif [ "$1" = "cycle" ]; then
	if [ "$2" = "vcs" ]; then
		echo "Running VCS"
		bash data-collection-vcs.sh
	elif [ "$2" = "verilator" ]; then
		echo "Running Verilator"
		bash data-collection-verilator.sh
	elif [ "$2" = "midas" ]; then
		echo "Running Midas"
		bash data-collection-midas.sh $3
	else
		echo "Invalid second parameter passed into gen-data.sh: should be 'vcs', 'verilator' or 'midas'"
		exit 1
	fi
fi

