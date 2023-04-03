
import gymnasium as gym
import miniworld
from miniworld.wrappers import PyTorchObsWrapper,GreyscaleWrapper
import numpy as np
from tqdm import tqdm

train_args = dict(
    nb_sections=5,
    proba_change_motor_gain=0,
    min_section_length=5,
    max_section_length=10,
    training=False,
    max_episode_steps=100,
)

env = gym.make('MiniWorld-TaskHallway-v0', 
              view="agent", render_mode=None,
              **train_args)
env = GreyscaleWrapper(env)
# env = PyTorchObsWrapper(env)

observations = []
observation, info = env.reset()

for _ in tqdm(range(100000)):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
    observations.append(observation)

env.close()

observations = np.stack(observations)
print(observations.shape)

with open('data/randomRollout.npy', 'wb') as f:
    a = np.save(f, observations)
