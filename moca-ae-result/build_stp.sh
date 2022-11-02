#!/bin/bash

python3 make_fair.py $1/gemmini-tests-workload-workload_a/uartlog A &&
python3 make_fair.py $1/gemmini-tests-workload-workload_b/uartlog B &&
python3 make_fair.py $1/gemmini-tests-workload-workload_c/uartlog C

