#!/bin/bash
# if [ ! $# != 2 ]; then
#     echo "Please provide a CSV path and systolic array size"
#     exit 1
# fi
CSV_PATH=$( realpath $1 )
FIRESIM_ROOT_DIR=/home/centos/firesim-dosa
cd $FIRESIM_ROOT_DIR/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/layers
python extract_data_csv.py --csv_loc $CSV_PATH
cd ..
rm -f $FIRESIM_ROOT_DIR/deploy/workloads/gemmini/*_tilings-baremetal
python gemmini_data_collection.py --pe_dim $2 --spad_size $3 --acc_size $4
cd ..
rm -rf ./build
./build.sh
mkdir -p $FIRESIM_ROOT_DIR/deploy/workloads/gemmini/
cp ./build/bareMetalC/*_tilings-baremetal $FIRESIM_ROOT_DIR/deploy/workloads/gemmini/
cd $FIRESIM_ROOT_DIR/deploy
python gemmini.py --pe_dim $2
