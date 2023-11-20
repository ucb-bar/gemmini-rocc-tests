import subprocess
import time
import pathlib
import glob
import sys

from parse_results import collect_results, add_to_csv

def eval(csv_path, pe_dim):
    # grep firesim workload logs
    firesim_results_path = "/home/centos/firesim-dosa/deploy/results-workload/"
    orig_results_dirs = glob.glob(firesim_results_path + "*/")

    # build workload and run firesim
    gemmini_rocc_path = pathlib.Path(__file__).parent.resolve()
    args = ["./eval_script.sh", csv_path, str(pe_dim), "2048", "2048"]
    p = subprocess.check_call(args, cwd=gemmini_rocc_path, stdout=None, stderr=None)

    # read newly created logs
    results_dirs = glob.glob(firesim_results_path + "*/")
    new_results_dirs = []
    print(new_results_dirs)
    for results_dir in results_dirs:
        if results_dir not in orig_results_dirs:
            print("Found new firesim results in", results_dir)
            new_results_dirs.append(results_dir)

    # # DEBUG
    # new_results_dirs = new_results_dirs[-1:]

    results = collect_results(new_results_dirs, "tiled ", collect_auto_tile=False, run_auto=False)
    add_to_csv(results, csv_path, "tiled", new_csv_path=csv_path)

    return results

if __name__ == "__main__":
    # csv_path = "/home/centos/firesim-dosa/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/batch.csv"
    assert(len(sys.argv) == 3)
    csv_path = sys.argv[1] 
    pe_dim = int(sys.argv[2])
    eval(csv_path, pe_dim)

