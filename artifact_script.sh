#!/bin/bash
cd /home/centos/firesim-dosa/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/layers
python extract_data_csv.py --csv_loc ../../artifact/$1/$2.csv
cd ..
rm -f ./build/bareMetalC/*_tilings-baremetal
python gemmini_data_collection.py
cd ..
./build.sh
cp ./build/bareMetalC/*_tilings-baremetal ~/firesim/deploy/workloads/gemmini/
cd /home/centos/firesim-dosa/deploy
python gemmini.py
