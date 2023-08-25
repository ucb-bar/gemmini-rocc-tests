#!/bin/bash
if [ ! $# -eq 2 ]; then
    echo "Please provide a predictor type and target workload"
    exit 1
fi
FIRESIM_ROOT_DIR=/home/centos/firesim-dosa
cd $FIRESIM_ROOT_DIR/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/layers
python extract_data_csv.py --csv_loc ../artifact_v2/$1/$2.csv
cd ..
rm -f $FIRESIM_ROOT_DIR/deploy/workloads/gemmini/*_tilings-baremetal
python gemmini_data_collection.py
cd ..
rm -rf ./build
./build.sh
mkdir -p $FIRESIM_ROOT_DIR/deploy/workloads/gemmini/
cp ./build/bareMetalC/*_tilings-baremetal $FIRESIM_ROOT_DIR/deploy/workloads/gemmini/
cd $FIRESIM_ROOT_DIR/deploy
python gemmini.py
