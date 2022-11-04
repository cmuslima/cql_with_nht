import sys
from nht.utils import action_map_arg_parser
from nht.train_action_map import train_action_map

arg_list = [
    "--max_epochs", "100", #100 is the typical number of training epochs
    "--d4rl_dset", "halfcheetah-expert-v2",
    "--dataset_prop", "0.5",
    #"--dataset_transitions", "10000", # overrides dataset_prop argument - comment out this line if you want to use percentage of data
    "--default_root_dir", ".results",
    "--lr", "0.001", #0.0001
    "--rnd_seed", "101",
    "--run_name", "test2",
    "--hiddens", "128,128,128",
    "--accelerator", "gpu",
    #"--accelerator", "cpu",
    "--devices", "1", # uncomment along with gpu line above to use gpu (also make sure to comment cpu line)
    "--a_dim", "2", #try 3 nextß
    "--context", "observations", 
    "--model", "NHT",
    "--multihead",
    "--lipschitz_coeff", "20", #next 1, 20, 100
    "--clip_action", "0.999",
]

sys.argv.extend(arg_list)


parser = action_map_arg_parser()
args = parser.parse_args()
args.run_name = f'dataset_percent_{args.dataset_prop}_{args.lipschitz_coeff}_{args.lr}'
args.default_root_dir = f'./trained_maps/{args.model}/{args.d4rl_dset}/action_dim_{args.a_dim}'
print(args)

    # train action map
model = train_action_map(args)