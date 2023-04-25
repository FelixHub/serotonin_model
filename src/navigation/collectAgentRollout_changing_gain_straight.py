# pylint: disable=W0611
import gymnasium as gym
import miniworld
import numpy as np
import torch
from agent import DEVICE, Policy, select_action
from miniworld.wrappers import GreyscaleWrapper, PyTorchObsWrapper
from tqdm import tqdm

args = dict(
    nb_sections=1,
    proba_change_motor_gain=0,
    min_section_length=25,
    max_section_length=40,
    training=True,
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
        if i_trajectories % 100 == 0:
            env = gym.make(
                "MiniWorld-TaskHallway-v0", view="agent", render_mode=None, **args
            )

            env = PyTorchObsWrapper(env)

        observations = []
        actions = []
        observation, _ = env.reset()

        # we get the steps where there is a gain change
        nb_changes = np.random.randint(1, 5)
        gain_change_steps = list(
            np.sort(np.random.choice(range(1, 10), size=nb_changes, replace=False) * 10)
        )
        gain_change_steps = [0] + gain_change_steps
        print(gain_change_steps)

        for i_step in range(args["max_episode_steps"]):
            if (len(gain_change_steps) > 0) and (i_step == gain_change_steps[0]):
                env.change_gain(motor_gains=[0.5, 1, 1.5])
                gain_change_steps.pop(0)

            # agent policy that uses the observation and info
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
        "data/rollout_changing_gain_straight/agentRollout_observations_"
        + str(id_run)
        + ".npy",
        "wb",
    ) as f:
        np.save(f, trajectories)
    with open(
        "data/rollout_changing_gain_straight/agentRollout_actions_"
        + str(id_run)
        + ".npy",
        "wb",
    ) as f:
        np.save(f, trajectories_action)

    del trajectories, trajectories_action, env
    torch.cuda.empty_cache()


i_run = 0
run_trajectories(nb_trajectories=10, id_run=i_run)
