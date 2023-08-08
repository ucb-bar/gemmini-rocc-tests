import glob
import pickle
import pandas as pd
import numpy as np

def collect_results(substrs, custom_str, collect_auto_tile=False):
    logs = glob.glob(f"/home/centos/firesim/deploy/results-workload/*/*/uartlog", recursive=True)
    relevant_logs = set()
    for substr in substrs:
        substr_logs = [log for log in logs if substr in log]
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
                    if custom_str == "tiled ":
                        row_str = lines[i-3][:-1]
                    elif collect_auto_tile:
                        row_str = lines[i-17][:-1]
                    else:
                        row_str = lines[i-1][:-1]
                    vals = row_str.split("_")
                    if line.startswith(f"Gemmini {custom_str}conv took"):
                        name_vals = vals[:7] + vals[8:]
                        name_vals.append("mapping")
                    elif line.startswith(f"Gemmini {custom_str}matmul took"):
                        name_vals = ["1", vals[0], vals[2], vals[1], "1",
                                     "1", "1"]
                        name_vals.extend(["1", "1", vals[3], "1", vals[4], vals[5], "1",
                                          vals[6], vals[7], vals[8]])
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
                                auto_tiling_factors.append(str(int(auto_tile_line[:-1].split(" = ")[-1]) * 16))
                            results[layer_name + "_auto_tiling"] = "_".join(auto_tiling_factors)
    return results

def add_to_csv(results, csv_path, col="", tile_only=False, new_csv_path=""):
    df = pd.read_csv(csv_path)
    with open("layers/layers.pickle", "rb") as f:
        layer_lst = pickle.load(f)
    if col == "auto": target_key = "target.gemmini_auto_cycle"
    elif col == "tiled": target_key = "target.gemmini_cycle"
    else: target_key = "target.gemmini_cycle"

    num_layers_found = 0
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
        # if layer["prob_name"] + "_auto_tiling" in results:
        #     df_idx = layer["df_idx"]
        #     df.loc[df_idx, "dse.auto_tiling"] = results[layer["prob_name"] + "_auto_tiling"]
    print(f"Found {len(layer_names)} unique layers")
    print(f"Found {num_layers_found} layers")
    df[target_key].replace('', np.nan, inplace=True)
    df.dropna(subset=[target_key], inplace=True)
    # df["target.cycle"] = df[target_key]
    if not new_csv_path:
        new_csv_path = csv_path
    df.to_csv(new_csv_path, index=False)

if __name__ == "__main__":
    # results = collect_results([
    #     "2023-04-12--07-39",
    #     "2023-04-12--06-21",
    #     "2023-04-12--05-15",
    #     "2023-04-12--04-08",
    #     "2023-04-12--02-56",
    #     "2023-04-12--01-25",
    #     "2023-04-17--19-11",
    #     "2023-04-17--19-35",
    #     "2023-04-17--2",
    # ])
    # results = collect_results([
    #     "2023-06-13--23",
    #     "2023-06-14",
    # ], "tiled ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/dla-dataset/data/dataset_1000map.csv", "tiled", new_csv_path="/home/centos/dla-dataset/data/dataset_1000map_firesim.csv")

    # results = collect_results([
    #     "2023-04-25--21-32-38",
    # ], "auto ", collect_auto_tile=True)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/gd_resnet50_4_22_0.csv")
    # results = collect_results([
    #     "2023-04-25--21-32-38",
    # ], "auto ", collect_auto_tile=True)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/gd_resnet50_4_22_0.csv")
    # results = collect_results([
    #     "2023-04-25--22-25-2",
    # ], "auto ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/gd_resnet50_4_22_0.csv", "auto")
    # print(sum(results.values()))

    # results = collect_results([
    #     "2023-04-25--22-25-2",
    # ], "tiled ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/gd_resnet50_4_22_0.csv", "tiled")

    # bert
    # results = collect_results([
    #     "2023-04-26--22-19-23",
    # ], "auto ", collect_auto_tile=True)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/tl_unet.csv", "auto", tile_only=True)

    # results = collect_results([
    #     "2023-04-26--23-04-41",
    # ], "auto ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/gd_unet_0.csv", "auto")
    # results = collect_results([
    #     "2023-04-26--23-04-41",
    # ], "tiled ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/gd_unet_0.csv", "tiled")
    # print(len(results))

    # 2023-04-26--04-26-29 - BERT TL
    # for name, cycles in sorted(results.items()):
    #     print(name, cycles)
    # print(sum(results.values()))

    # # model only
    # # bert
    # results = collect_results([
    #     "2023-07-08--13-35-52",
    # ], "auto ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/model_only/bert.csv", "auto")
    # results = collect_results([
    #     "2023-07-08--13-35-52",
    # ], "tiled ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/model_only/bert.csv", "tiled")

    # # unet
    # results = collect_results([
    #     "2023-07-08--11-43-09",
    # ], "auto ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/model_only/unet.csv", "auto")
    # results = collect_results([
    #     "2023-07-08--11-43-09",
    # ], "tiled ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/model_only/unet.csv", "tiled")

    # # retinanet
    # results = collect_results([
    #     "2023-07-08--12-28-44",
    # ], "auto ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/model_only/retinanet.csv", "auto")
    # results = collect_results([
    #     "2023-07-08--12-28-44",
    # ], "tiled ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/model_only/retinanet.csv", "tiled")

    # # resnet50
    # results = collect_results([
    #     "2023-07-08--12-40-03",
    #     "2023-07-08--12-43-24",
    # ], "auto ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/model_only/resnet50.csv", "auto")
    # results = collect_results([
    #     "2023-07-08--12-40-03",
    #     "2023-07-08--12-43-24",
    # ], "tiled ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/gemmini-data-collection/model_only/resnet50.csv", "tiled")

    # analytical + model
    # results = collect_results([
    #     "2023-08-03--23-10-35",
    #     "2023-08-03--23-08-47",
    # ], "tiled ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/artifact/resnet50.csv", "tiled")
    # results = collect_results([
    #     "2023-08-03--23-10-35",
    #     "2023-08-03--23-08-47",
    # ], "auto ", collect_auto_tile=False)
    # add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/artifact/resnet50.csv", "auto")

    results = collect_results([
        "2023-08-05--00",
        "2023-08-05--01-00",
    ], "tiled ", collect_auto_tile=False)
    add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/artifact/resnet50_1.csv", "tiled")
    results = collect_results([
        "2023-08-05--00",
        "2023-08-05--01-00",
    ], "auto ", collect_auto_tile=False)
    add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/artifact/resnet50_1.csv", "auto")

    results = collect_results([
        "2023-08-05--01-2",
    ], "tiled ", collect_auto_tile=False)
    add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/artifact/resnet50_2.csv", "tiled")
    results = collect_results([
        "2023-08-05--01-2",
    ], "auto ", collect_auto_tile=False)
    add_to_csv(results, "/home/centos/firesim-esp/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/artifact/resnet50_2.csv", "auto")
