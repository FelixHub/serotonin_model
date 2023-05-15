
import gymnasium as gym
import miniworld
from miniworld.wrappers import PyTorchObsWrapper,GreyscaleWrapper
import numpy as np
from tqdm import tqdm

rollout_args = dict(
    min_section_length=10,
    max_section_length=50,
    max_episode_steps=100,
    facing_forward=True,
    reset_keep_same_length=False,
    wall_tex='stripe_gradient', #  stripes_big
)

env = gym.make('MiniWorld-TaskHallwaySimple-v0', 
              view="agent", render_mode=None,
              **rollout_args)
env = GreyscaleWrapper(env)
# env = PyTorchObsWrapper(env)

observations = []
observation, info = env.reset()

for i in tqdm(range(100000)):
    
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
    observations.append(observation)

env.close()

observations = np.stack(observations)
print(observations.shape)

with open('../data/navigation/randomRollout_alt_texture.npy', 'wb') as f:
    a = np.save(f, observations)

