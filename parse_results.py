import glob
import pickle
import argparse

import pandas as pd
import numpy as np

def collect_results(substrs, custom_str, collect_auto_tile=False):
    # logs = glob.glob(f"/home/centos/firesim/deploy/results-workload/*/*/uartlog", recursive=True)
    relevant_logs = set()
    # for substr in substrs:
    #     substr_logs = [log for log in logs if substr in log]
    #     for log in substr_logs:
    #         relevant_logs.add(log)
    for substr in substrs:
        substr_logs = glob.glob(f"{substr}/*/uartlog", recursive=True)
        for log in substr_logs:
            relevant_logs.add(log)
    print(relevant_logs)

    results = {}
    for log in relevant_logs:
        # layer_name = log.split("/")[-2].split("-baremetal")[0]
        with open(log, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith(f"Gemmini {custom_str}conv took") or line.startswith(f"Gemmini {custom_str}matmul took"):
                    # import pdb
                    # pdb.set_trace()
                    # if custom_str == "tiled ":
                    #     row_str = lines[i-3][:-1]
                    # elif collect_auto_tile:
                    #     row_str = lines[i-17][:-1]
                    # else:
                    #     row_str = lines[i-1][:-1]
                    if "matmul" in line:
                        if custom_str == "tiled ":
                            row_str = lines[i-13]
                        else:
                            row_str = lines[i-11]
                    if "conv" in line:
                        if custom_str == "tiled ":
                            row_str = lines[i-19]
                        else:
                            row_str = lines[i-17]
                    vals = row_str.split("_")
                    if line.startswith(f"Gemmini {custom_str}conv took"):
                        name_vals = vals[:7] + vals[8:]
                        name_vals[-1] = name_vals[-1][:-1]
                        name_vals.append("mapping")
                    elif line.startswith(f"Gemmini {custom_str}matmul took"):
                        name_vals = ["1", vals[0], vals[2], vals[1], "1",
                                     "1", "1"]
                        name_vals.extend(["1", "1", vals[3], "1", vals[4], vals[5], "1",
                                          vals[6], vals[7], vals[8][:-1]])
                        name_vals.append("mapping")
                    layer_name = "_".join(name_vals)
                    cycles = int(line.split(" ")[-2])
                    if not collect_auto_tile:
                        results[layer_name] = cycles
                    if collect_auto_tile:
                        if "conv" in line:
                            auto_tiling_factors = []
                            for auto_tile_line in [lines[i-11], lines[i-12], lines[i-14], lines[i-15], lines[i-10], lines[i-13], lines[i-16]]:
                                auto_tiling_factors.append(auto_tile_line[:-1].split(" = ")[-1])
                            results[layer_name + "_auto_tiling"] = "_".join(auto_tiling_factors)
                        elif "matmul" in line:
                            auto_tiling_factors = []
                            for auto_tile_line in [1, 1, lines[i-10], 1, lines[i-8], lines[i-9], 1]:
                                if isinstance(auto_tile_line, int):
                                    auto_tiling_factors.append(str(auto_tile_line))
                                else:
                                    auto_tiling_factors.append(str(int(auto_tile_line[:-1].split(": ")[-1]) * 16))
                            results[layer_name + "_auto_tiling"] = "_".join(auto_tiling_factors)
    return results

def add_to_csv(results, csv_path, col="", tile_only=False, new_csv_path=""):
    df = pd.read_csv(csv_path)
    with open("gemmini-data-collection/layers/layers.pickle", "rb") as f:
        layer_lst = pickle.load(f)
    target_key = "dse.auto_tiling"
    if col == "auto": target_key = "target.gemmini_auto_cycle"
    elif col == "tiled": target_key = "target.gemmini_cycle"

    num_layers_found = 0
    num_tilings_found = 0
    layer_names = dict()
    for layer in layer_lst:
        if not tile_only and layer["prob_name"] in results:
            if layer["prob_name"] in layer_names:
                existing_cycle = df["target.cycle"][layer_names[layer["prob_name"]]["df_idx"]]
                new_cycle = df["target.cycle"][layer["df_idx"]]
                if new_cycle != existing_cycle:
                    import pdb
                    pdb.set_trace()
                continue
            layer_names[layer["prob_name"]] = layer
            num_layers_found += 1
            df_idx = layer["df_idx"]
            df.loc[df_idx, target_key] = results[layer["prob_name"]]
        if layer["prob_name"] + "_auto_tiling" in results:
            df_idx = layer["df_idx"]
            df.loc[df_idx, "dse.auto_tiling"] = results[layer["prob_name"] + "_auto_tiling"]
            num_tilings_found += 1
    print(f"Found {len(layer_names)} unique layers")
    print(f"Found {num_layers_found} layers")
    print(f"Found {num_tilings_found} tilings")
    df[target_key].replace('', np.nan, inplace=True)
    df.dropna(subset=[target_key], inplace=True)
    # df["target.cycle"] = df[target_key]
    if not new_csv_path:
        new_csv_path = csv_path
    df.to_csv(new_csv_path, index=False)

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')

    parser.add_argument('--result',
                        type=str,
                        help='FireSim result dir(s)',
                        action='append',
                        )
    parser.add_argument('-wl',
                        '--workload',
                        type=str,
                        help='<Required> Name of workload directory.',
                        required=True,
                        )
    parser.add_argument('--pred',
                        type=str,
                        help='Predictor type (analytical|both|dnn)',
                        required=True,
                        )
    return parser

if __name__ == "__main__":
    args = construct_argparser().parse_args()

    results = collect_results(args.result, "auto ", collect_auto_tile=True)
    print(results)
    add_to_csv(results, f"gemmini-data-collection/artifact/{args.pred}/{args.workload}.csv", "auto", tile_only=True, new_csv_path="test.csv")
    results = collect_results(args.result, "auto ", collect_auto_tile=False)
    print(results)
    add_to_csv(results, f"test.csv", "auto", new_csv_path="test.csv")
    # add_to_csv(results, f"gemmini-data-collection/artifact/{args.pred}/{args.workload}.csv", "auto", new_csv_path="test.csv")
    results = collect_results(args.result, "tiled ", collect_auto_tile=False)
    print(results)
    add_to_csv(results, f"test.csv", "tiled", new_csv_path="test.csv")
    # add_to_csv(results, f"gemmini-data-collection/artifact/{args.pred}/{args.workload}.csv", "tiled", new_csv_path="test.csv")
