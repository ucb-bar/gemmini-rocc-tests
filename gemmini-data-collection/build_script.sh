#!/bin/bash
if [ ! $# -eq 1 ]; then
    # echo "Please provide a CSV path and systolic array size"
    echo "Please provide a CSV path"
    exit 1
fi
CSV_PATH=$( realpath $1 )
cd layers
python extract_data_csv.py --csv_loc $CSV_PATH
cd ..
python gemmini_data_collection.py --pe_dim 16
cd ..
rm -rf ./build
./build.sh
