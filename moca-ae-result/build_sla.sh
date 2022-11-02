#!/bin/bash

python3 parse_result_from_uartlog.py $1/gemmini-tests-workload-workload_a/uartlog A &&
python3 parse_result_from_uartlog.py $1/gemmini-tests-workload-workload_b/uartlog B &&
python3 parse_result_from_uartlog.py $1/gemmini-tests-workload-workload_c/uartlog C

