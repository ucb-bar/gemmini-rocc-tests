#!/bin/bash

cd ../../..
sed -i "/val customConfig =/c\ val customConfig = $1" src/main/scala/gemmini/CustomConfigs.scala
sed -i "/val customConfig =/c\ val customConfig = $1" configs/GemminiCustomConfigs.scala
if [ "$2" = "vcs" ]; then
	./scripts/build-vcs.sh
elif [ "$2" = "verilator" ]; then
	./scripts/build-verilator.sh
elif [ "$2" = "midas" ]; then
	./scripts/build-midas.sh $3
else
	echo "Invalid second paramter passed into gen-data.sh: should be 'vcs', 'verilator' or 'midas'"
	exit 1
fi

./scripts/build-spike.sh
cd software/gemmini-rocc-tests/gemmini-data-collection

result_dir=../../../data-collection-output
tiling_dir=../../../data-collection-output-configs/data-collection-output-tiling-factors-$1
cycle_dir=../../../data-collection-output-configs/data-collection-output-cycles-$2-$1

mkdir -p $tiling_dir
mkdir -p $cycle_dir

bash gen_data.sh tile
mv $result_dir/* $tiling_dir/ && rmdir $result_dir
bash gen_data.sh cycle $2 $3
mv $result_dir/* $cycle_dir/ && rmdir $result_dir

