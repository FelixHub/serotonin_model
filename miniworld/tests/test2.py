import gymnasium as gym
import miniworld
env = gym.make('MiniWorld-OneRoomS6-v0', view="agent", render_mode="human")
observation, info = env.reset()

# Create the display window
env.render()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        observation, info = env.reset()
    

env.close()