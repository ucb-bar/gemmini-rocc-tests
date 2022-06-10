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
        compression_factor = 1 if num_heads == 4 else 2

        print(
f"""    PRINT_ENCODER_DECODER("sehoon-{i}-enc-{j}", /*is_encoder=*/true,
            /*hidden_dim=*/{hidden_dim}, /*expansion_dim=*/{expansion_dim}, /*num_heads=*/{num_heads}, /*cross_num_heads=*/{num_heads}, /*seq_len=*/{seq_len}, /*compression_factor=*/{compression_factor});
""")

    for j,decoder in enumerate(decoders):
        expansion_dim, num_heads, cross_num_heads = decoder
        compression_factor = 1 if num_heads == 4 else 2

        print(
f"""    PRINT_ENCODER_DECODER("sehoon-{i}-dec-{j}", /*is_encoder=*/false,
            /*hidden_dim=*/{hidden_dim}, /*expansion_dim=*/{expansion_dim}, /*num_heads=*/{num_heads}, /*cross_num_heads=*/{cross_num_heads}, /*seq_len=*/{seq_len}, /*compression_factor=*/{compression_factor});
""")

    print()

