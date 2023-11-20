import pickle, argparse
import yaml
import pandas as pd

if __name__ == "__main__":
    # with open("layers.yaml", "r") as f:
    #     layer_dict = yaml.safe_load(f)
    # names = set()
    # for entry in layer_dict:
    #     names.add(entry["prob_name"])
    # print(len(names))
    # exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_loc", type=str, required=True)
    args = parser.parse_args()

    layer_dicts = []
    df = pd.read_csv(args.csv_loc)
    for index, row in df.iterrows():
        layer_dict = {}
        dims = ("R", "S", "P", "Q", "C", "K", "N", "Wstride", "Hstride", "Wdilation", "Hdilation")
        # layer_dict["prob_name"] = "_".join([str(row[f"prob.{dim}"]) for dim in dims])
        layer_dict["BATCH_SIZE"] = row["prob.N"]
        layer_dict["IN_CHANNELS"] = row["prob.C"]
        layer_dict["OUT_CHANNELS"] = row["prob.K"]
        layer_dict["KERNEL_DIM"] = row["prob.R"]
        layer_dict["STRIDE"] = row["prob.Wstride"]
        layer_dict["KERNEL_DILATION"] = row["prob.Wdilation"]
        if row["prob.Q"] == 1 and row["prob.R"] == 1 and row["prob.S"] == 1: # this is a matmul
            layer_dict["I"] = row["prob.P"] 
            layer_dict["K"] = row["prob.C"]
            layer_dict["J"] = row["prob.K"]
            layer_dict["IN_DIM"] = row["prob.P"]
            layer_dict["OUT_DIM"] = row["prob.P"]
        else:
            # layer_dict["IN_DIM"] = (row["prob.P"] - 1) * layer_dict["STRIDE"] + 2 * layer_dict["KERNEL_DILATION"] + layer_dict["KERNEL_DIM"]
            # layer_dict["OUT_DIM"] = int((layer_dict["IN_DIM"] - layer_dict["KERNEL_DIM"]) / layer_dict["STRIDE"] + 1)
            layer_dict["IN_DIM"] = row["prob.P"] * layer_dict["STRIDE"]
            layer_dict["OUT_DIM"] = int((layer_dict["IN_DIM"] - layer_dict["KERNEL_DIM"] + layer_dict["KERNEL_DIM"] // 2 * 2) / layer_dict["STRIDE"] + 1)
        mapping_str = row["mapping.mapping"]
        factors = {}
        for lvl_str in ["L0[W] ", "L1[O] ", "L2[WI] "]:
            words = mapping_str.split(lvl_str)[1].split(" - ")[0].split(" ")
            for word in words:
                if "X" in word or "Y" in word:
                    word = word[:-1]
                    factors[word[0]+"X"] = int(word[1:])
                elif word[0] not in factors:
                    factors[word[0]] = int(word[1:])
                else:
                    factors[word[0]] *= int(word[1:])
        layer_dict["TILE_BATCHES"] = factors.get("N", 1)
        layer_dict["TILE_OCOLS"] = factors.get("P", 1)
        layer_dict["TILE_OROWS"] = factors.get("Q", 1)
        layer_dict["TILE_OCHS"] = factors.get("K", 1)
        layer_dict["TILE_KCOLS"] = factors.get("R", 1)
        layer_dict["TILE_KROWS"] = factors.get("S", 1)
        layer_dict["TILE_KCHS"] = factors.get("C", 1)
        
        layer_dict["TILE_KCHS"] *= factors.get("CX", 1)
        layer_dict["SPATIAL_TILE_KCHS"] = factors.get("CX", 1)
        layer_dict["TILE_OCHS"] *= factors.get("KX", 1)
        layer_dict["SPATIAL_TILE_OCHS"] = factors.get("KX", 1)

        # # scratchpad level K spatial
        # layer_dict["SPATIAL_TILE_OCHS"] = 1
        # words = mapping_str.split("L2[WI] ")[1].split(" - ")[0].split(" ")
        # for word in words:
        #     if ("K" in word) and ("X" in word):
        #         spatial_k = int(word[1:-1])
        #         layer_dict["TILE_OCHS"] *= spatial_k
        #         layer_dict["SPATIAL_TILE_OCHS"] = spatial_k

        # get reg level tiling factor
        words = mapping_str.split("L0[W] ")[1].split(" - ")[0].split(" ")
        factors = {}
        for word in words:
            factors[word[0]] = int(word[1:])
        layer_dict["TILE_OCOLS"] *= factors.get("P", 1)
        layer_dict["TILE_OROWS"] *= factors.get("Q", 1)

        layer_dicts.append(layer_dict)

        # gemmini default perm - CRSKPQN
        words = mapping_str.split("L3[WIO] ")[1].split(" - ")[0].split(" ")
        perm_str = "".join(reversed([word[0] for word in words]))
        for dim in "CRSKPQN":
            if dim not in perm_str:
                perm_str = perm_str + dim
        layer_dict["PERM_STR"] = '"' + perm_str + '"'
        # layer_dict["PERM_STR"] = perm_str

        layer_dict_dims = ["KERNEL_DIM", "OUT_DIM", "IN_CHANNELS", "OUT_CHANNELS", "BATCH_SIZE", "STRIDE", "KERNEL_DILATION"]
        mapping_keys = ["TILE_KCOLS", "TILE_KROWS", "TILE_OCOLS", "TILE_OROWS", "TILE_KCHS", "TILE_OCHS", "TILE_BATCHES", "SPATIAL_TILE_KCHS", "SPATIAL_TILE_OCHS"]
        mapping_lst = map(str, [layer_dict[k] for k in mapping_keys])
        # layer_dict["prob_name"] = "_".join([str(row[f"prob.{dim}"]) for dim in dims]) + "_" + "_".join(mapping_lst) + "_" + perm_str + "_mapping"
        layer_dict["prob_name"] = "_".join([str(int(layer_dict[dim])) for dim in layer_dict_dims]) + "_" + "_".join(mapping_lst) + "_" + perm_str + "_mapping"
        layer_dict["df_idx"] = index

    with open('layers.pickle', 'wb') as p:
        pickle.dump(layer_dicts, p, protocol=pickle.HIGHEST_PROTOCOL)
    with open("layers.yaml", "w") as f:
        yaml.dump(layer_dicts, f)

    #sample: python extract_data.py --layer_file resnet50/unique_layers.yaml --prob_loc resnet50/ --map_loc resnet50_map_v2/logs/gemmini_16_256.0_64.0/
