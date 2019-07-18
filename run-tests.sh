#!/usr/bin/env bash

echo "*****************TEST RESULTS*************" > test_output.txt
echo "=========tiled_matmul_os-linux=========" >> test_output.txt
/root/systolic-rocc-tests/build/bareMetalC/tiled_matmul_os-linux >> test_output.txt
echo "=========tiled_matmul_ws-linux=========" >> test_output.txt
/root/systolic-rocc-tests/build/bareMetalC/tiled_matmul_ws-linux >> test_output.txt
echo "========mobilenet=========" >> test_output.txt
/root/systolic-rocc-tests/build/mobilenet/test >> test_output.txt
echo "========mobilenet=========" >> test_output.txt
/root/systolic-rocc-tests/build/mobilenet/test cpu >> test_output.txt
cat test_output.txt
