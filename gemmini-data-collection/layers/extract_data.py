import pickle, yaml, os, re, argparse

def extract_prob(layer_dict, prob):
    assert prob["Wstride"] == prob["Hstride"]
    assert prob["Wdilation"] == prob["Hdilation"]
    assert prob["P"] == prob["Q"]
    assert prob["R"] == prob["S"]

    layer_dict["BATCH_SIZE"] = prob["N"] 
    layer_dict["IN_CHANNELS"] = prob["C"]
    layer_dict["OUT_CHANNELS"] = prob["K"]
    layer_dict["KERNEL_DIM"] = prob["R"]
    layer_dict["STRIDE"] = prob["Wstride"]
    layer_dict["KERNEL_DILATION"] = prob["Wdilation"]
    layer_dict["IN_DIM"] = (prob["P"] - 1) * layer_dict["STRIDE"] + 2 * layer_dict["KERNEL_DILATION"] + layer_dict["KERNEL_DIM"] 
    

def extract_mapping(layer_dict, prob, loc):
    map_dir = "{R}_{S}_{P}_{Q}_{C}_{K}_{N}_{Wstride}_{Hstride}_{Wdilation}_{Hdilation}".format(R=prob["R"], S=prob["S"], P=prob["P"], Q=prob["Q"], C=prob["C"], K=prob["K"], N=prob["N"], Wstride=prob["Wstride"], Hstride=prob["Hstride"], Wdilation=prob["Wdilation"], Hdilation=prob["Hdilation"])
    map_filename = os.path.join(loc, map_dir, "timeloop-mapper.map.yaml")
    layer_dict["prob_name"] = map_dir

    with open(map_filename) as f:
        f_text = f.read() 

    def tiling_factor(letter):
        p = re.compile("\s{l}(\d+)\s".format(l=letter))
        factors = p.findall(f_text)
        product = 1
        for num in factors:
            product *= int(num)
        product /= int(factors[-1]) #remove DRAM factor
        product /= int(factors[-2]) #remove spad temporal
        product /= int(factors[-3]) #remove spad spatial
        return product

    layer_dict["TILE_BATCHES"] = tiling_factor("N")
    layer_dict["TILE_OCOLS"] = tiling_factor("Q")
    layer_dict["TILE_OROWS"] = tiling_factor("P")
    layer_dict["TILE_OCHS"] = tiling_factor("K")
    layer_dict["TILE_KCOLS"] = tiling_factor("S")
    layer_dict["TILE_KROWS"] = tiling_factor("R")
    layer_dict["TILE_KCHS"] = tiling_factor("C")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_file", type=str, required=True)
    parser.add_argument("--map_loc", type=str, required=True)
    parser.add_argument("--prob_loc", type=str, required=True)
    args = parser.parse_args()

    layer_dicts = []
    with open(args.layer_file) as f:
        layers = yaml.safe_load(f)
    for layer in layers:
        layer_filename = os.path.join(args.prob_loc, layer + ".yaml")
        with open(layer_filename) as f:
            prob = yaml.safe_load(f)["problem"]
        layer_dict = {}
        extract_prob(layer_dict, prob)
        extract_mapping(layer_dict, prob, args.map_loc)
        layer_dicts.append(layer_dict)

    with open('layers.pickle', 'wb') as p:
        pickle.dump(layer_dicts, p, protocol=pickle.HIGHEST_PROTOCOL) 

    #sample: python extract_data.py --layer_file resnet50/unique_layers.yaml --prob_loc resnet50/ --map_loc resnet50_map_v2/logs/gemmini_16_256.0_64.0/
