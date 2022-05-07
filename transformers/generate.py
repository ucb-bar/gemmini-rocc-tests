import json

configs = json.load(open("configs.json"))

hidden_dim = 512
seq_len = 128

def unique(l):
    return list(dict.fromkeys(list(l)))

for i,config in enumerate(configs):
    config = config['config']

    encoders = unique(zip(config['encoder']['fc'], config['encoder']['head']))
    decoders = unique(zip(config['decoder']['fc'], config['decoder']['head'], config['decoder']['xhead']))

    for j,encoder in enumerate(encoders):
        expansion_dim, num_heads = encoder

        print(
f"""    PRINT_ENCODER("sehoon-{i}-enc-{j}",
            /*hidden_dim=*/{hidden_dim}, /*expansion_dim=*/{expansion_dim}, /*num_heads=*/{num_heads}, /*seq_len=*/{seq_len});
""")

    for j,decoder in enumerate(decoders):
        expansion_dim, num_heads, cross_num_heads = decoder

        print(
f"""    PRINT_DECODER("sehoon-{i}-dec-{j}",
            /*hidden_dim=*/{hidden_dim}, /*expansion_dim=*/{expansion_dim}, /*num_heads=*/{num_heads}, /*cross_num_heads=*/{cross_num_heads}, /*seq_len=*/{seq_len});
""")

    print()

