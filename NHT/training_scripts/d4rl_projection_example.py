from nht.utils import project_d4rl_actions


model = 'NHT'
d4rl_dset = "halfcheetah-expert-v2"
prop = 1.0
NHT_path = f'./trained_maps/{model}/{d4rl_dset}/dataset_percent_{prop}/version_0'
dset_with_projected_actions = project_d4rl_actions(NHT_path, d4rl_dset, prop=0.01)

for i in range(5):
   
    a = dset_with_projected_actions['actions'][i]
    print(a)

print(dset_with_projected_actions.keys())