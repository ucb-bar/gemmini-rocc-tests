#!/bin/bash

cd ../../..
sed -i "/val customConfig =/c\ val customConfig = $1" configs/GemminiCustomConfigs.scala
./scripts/build-vcs.sh
./scripts/build-spike.sh
cd software/gemmini-rocc-tests/gemmini-data-collection
bash gen_data.sh tile
mkdir -p ../../../data-collection-output-configs 
mv ../../../data-collection-output ../../../data-collection-output-configs/data-collection-output-spike-$1
bash gen_data.sh
mv ../../../data-collection-output ../../../data-collection-output-configs/data-collection-output-vcs-$1
