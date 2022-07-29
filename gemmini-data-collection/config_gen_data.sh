#!/bin/bash

cd ../../..
sed -i "/val customConfig =/c\ val customConfig = $1" src/main/scala/gemmini/CustomConfigs.scala
sed -i "/val customConfig =/c\ val customConfig = $1" configs/GemminiCustomConfigs.scala 
if [ "$2" = "vcs" ]; then
	./scripts/build-vcs.sh
elif [ "$2" = "midas" ]; then
	./scripts/build-midas.sh
else
	echo "Invalid second paramter passed into gen-data.sh: should be 'vcs' or 'midas'"
	exit 1
fi
./scripts/build-spike.sh
cd software/gemmini-rocc-tests/gemmini-data-collection
bash gen_data.sh tile $2
mkdir -p ../../../data-collection-output-configs 
mv ../../../data-collection-output ../../../data-collection-output-configs/data-collection-output-spike-$1
bash gen_data.sh cycle $2
mv ../../../data-collection-output ../../../data-collection-output-configs/data-collection-output-$2-$1
