#!/usr/bin/env bash

echo "*****************TEST RESULTS*************" > test_output.txt
echo "=========tiled_matmul========="
echo "=========tiled_matmul_os-linux=========" >> test_output.txt
/root/gemmini-rocc-tests/build/bareMetalC/tiled_matmul_os-linux >> test_output.txt
echo "=========tiled_matmul_ws-linux=========" >> test_output.txt
/root/gemmini-rocc-tests/build/bareMetalC/tiled_matmul_ws-linux >> test_output.txt
echo "=========tiled_matmul_cpu-linux=========" >> test_output.txt
/root/gemmini-rocc-tests/build/bareMetalC/tiled_matmul_cpu-linux >> test_output.txt
echo "========mobilenet========="
echo "========mobilenet OS=========" >> test_output.txt
/root/gemmini-rocc-tests/build/mobilenet/mobilenet32 os >> test_output.txt
echo "========mobilenet WS=========" >> test_output.txt
/root/gemmini-rocc-tests/build/mobilenet/mobilenet32 ws >> test_output.txt
echo "========mobilenet CPU=========" >> test_output.txt
/root/gemmini-rocc-tests/build/mobilenet/mobilenet32 cpu >> test_output.txt
echo "========Gemmini Library Test 5========="
echo "========Gemmini Library Test 5 OS=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test5 os >> test_output.txt
echo "========Gemmini Library Test 5 WS=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test5 ws >> test_output.txt
echo "========Gemmini Library Test 5 CPU=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test5 cpu >> test_output.txt
echo "========Gemmini Library Test 6========="
echo "========Gemmini Library Test 6 OS=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test6 os >> test_output.txt
echo "========Gemmini Library Test 6 WS=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test6 ws >> test_output.txt
echo "========Gemmini Library Test 6 CPU=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test6 cpu >> test_output.txt
echo "========Gemmini Library Test 7========="
echo "========Gemmini Library Test 7 OS=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test7 os >> test_output.txt
echo "========Gemmini Library Test 7 WS=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test7 ws >> test_output.txt
echo "========Gemmini Library Test 7 CPU=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test7 cpu >> test_output.txt
echo "========Gemmini Library Test 8========="
echo "========Gemmini Library Test 8 OS=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test8 os >> test_output.txt
echo "========Gemmini Library Test 8 WS=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test8 ws >> test_output.txt
echo "========Gemmini Library Test 8 CPU=========" >> test_output.txt
/root/gemmini-rocc-tests/build/gemmini_library/test8 cpu >> test_output.txt
cat test_output.txt
poweroff -f
