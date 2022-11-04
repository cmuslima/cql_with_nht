from nht.mujoco_interface import register_NHT_env
import gym


model = 'NHT'
d4rl_dset = "halfcheetah-medium-expert-v2"
NHT_path = '/Users/cmuslimani/Projects/action_mapping/NHT/training_scripts/trained_maps/NHT/halfcheetah-medium-expert-v2/dataset_percent_0.5/version_0'

env = 'HalfCheetah-v2'

register_NHT_env(env, NHT_path) # gym registration
my_NHT_env = gym.make('NHT_HalfCheetah-v2')

# print some info about the environment to see that it works
my_NHT_env.reset()
print(f'Action space: {my_NHT_env.action_space}')
print(f'Obs space: {my_NHT_env.observation_space}')
print(f'NHT model \n{my_NHT_env.Q}') # NHT model