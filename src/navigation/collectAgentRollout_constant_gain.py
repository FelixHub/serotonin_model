# pylint: disable=W0611
import gymnasium as gym
import miniworld
import numpy as np
import torch
from agent import DEVICE, Policy, select_action
from miniworld.wrappers import GreyscaleWrapper, PyTorchObsWrapper
from tqdm import tqdm

args = dict(
    nb_sections=3,
    proba_change_motor_gain=0,
    min_section_length=3,
    max_section_length=8,
    training=False,
    max_episode_steps=100,
)


loaded_policy = Policy().to(DEVICE)
loaded_policy.load_state_dict(torch.load("saved_models/miniworld_task.pt"))
loaded_policy.eval()


def run_trajectories(nb_trajectories, id_run):
    env = gym.make("MiniWorld-TaskHallway-v0", view="agent", render_mode=None, **args)
    env = PyTorchObsWrapper(env)

    trajectories = []
    trajectories_action = []
    nb_rewards = 0

    for i_trajectories in tqdm(range(nb_trajectories)):
        # we change environment every 10 trials
        if i_trajectories % 50 == 0:
            env = gym.make(
                "MiniWorld-TaskHallway-v0", view="agent", render_mode=None, **args
            )
            env = PyTorchObsWrapper(env)

        observations = []
        actions = []
        observation, _ = env.reset()

        for _ in range(args["max_episode_steps"]):
            action, _ = select_action(loaded_policy, observation)

            # we save s_t and a_t
            actions.append(action)
            observations.append(observation)

            observation, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
                if reward > 0:
                    nb_rewards += 1

        observations = np.stack(observations)
        actions = np.stack(actions)

        trajectories.append(observations)
        trajectories_action.append(actions)

    print("proportion of rewarded trials :", nb_rewards / nb_trajectories)

    trajectories = np.stack(trajectories)
    trajectories_action = np.stack(trajectories_action)

    # convert to grayscale
    trajectories = np.sum(trajectories, axis=-3, keepdims=1) / 3

    env.close()

    with open(
        "data/rollout_constant_gain/agentRollout_observations_" + str(id_run) + ".npy",
        "wb",
    ) as f:
        np.save(f, trajectories)
    with open(
        "data/rollout_constant_gain/agentRollout_actions_" + str(id_run) + ".npy", "wb"
    ) as f:
        np.save(f, trajectories_action)

    del trajectories, trajectories_action, env
    torch.cuda.empty_cache()


i_run = 4
run_trajectories(nb_trajectories=1000, id_run=i_run)
