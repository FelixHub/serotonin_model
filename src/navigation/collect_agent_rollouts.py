# pylint: disable=W0611, W0613
import gymnasium as gym
import miniworld
import numpy as np
import torch
from miniworld.wrappers import GreyscaleWrapper, PyTorchObsWrapper
from tqdm import tqdm

from .agent import DEVICE, Policy, select_action

loaded_policy = Policy().to(DEVICE)
loaded_policy.load_state_dict(torch.load("../saved_models/miniworld_task.pt"))
loaded_policy.eval()


def constant_gain_strategy(env, args, i_step, gain_change_steps):
    pass


def changing_gain_straight_strategy(env, args, i_step, gain_change_steps):
    if (len(gain_change_steps) > 0) and (i_step == gain_change_steps[0]):
        env.change_gain(motor_gains=[0.5, 1, 1.5])
        gain_change_steps.pop(0)


def changing_gain_straight_glitch_strategy(env, args, i_step, gain_change_steps):
    if (len(gain_change_steps) > 0) and (i_step == gain_change_steps[0]):
        env.change_gain(
            motor_gains=[0.5, 1, 1.5],
            glitch=True,
            glitch_phase=np.random.choice(
                [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
            ),
        )
        gain_change_steps.pop(0)


def run_trajectories(nb_trajectories, id_run, collect_type):
    if collect_type == "constant_gain":
        args = dict(
            nb_sections=3,
            proba_change_motor_gain=0,
            min_section_length=3,
            max_section_length=8,
            training=False,
            max_episode_steps=100,
        )
        strategy = constant_gain_strategy

    elif collect_type == "changing_gain_straight":
        args = dict(
            nb_sections=1,
            proba_change_motor_gain=0,
            min_section_length=25,
            max_section_length=40,
            training=True,
            max_episode_steps=100,
        )
        strategy = changing_gain_straight_strategy

    elif collect_type == "changing_gain_straight_glitch":
        args = dict(
            nb_sections=1,
            proba_change_motor_gain=0,
            min_section_length=25,
            max_section_length=40,
            training=True,
            max_episode_steps=100,
        )
        strategy = changing_gain_straight_glitch_strategy

    else:
        raise ValueError("Invalid collect_type")

    # create the environment and wrap it
    env = gym.make("MiniWorld-TaskHallway-v0", view="agent", render_mode=None, **args)
    env = PyTorchObsWrapper(env)

    # initialize (global) data storage variables
    trajectories = []
    trajectories_action = []
    nb_rewards = 0

    # for each trajectory (rollout), do the following
    for i_trajectories in tqdm(range(nb_trajectories)):
        # sloppy way to randomize trajectory length
        if i_trajectories % 50 == 0:
            env = gym.make(
                "MiniWorld-TaskHallway-v0", view="agent", render_mode=None, **args
            )
            env = PyTorchObsWrapper(env)

        # initialize data storage variables for each trajectory
        observations = []
        actions = []
        observation, _ = env.reset()
        gain_change_steps = []

        # depending on the strategy type, implement gain change
        if collect_type in ["changing_gain_straight", "changing_gain_straight_glitch"]:
            nb_changes = np.random.randint(1, 5)
            gain_change_steps = list(
                np.sort(
                    np.random.choice(range(1, 10), size=nb_changes, replace=False) * 10
                )
            )
            gain_change_steps = [0] + gain_change_steps

        # for each step in the trajectory, do the following
        for i_step in range(args["max_episode_steps"]):
            # strategy-specific initialization
            strategy(env, args, i_step, gain_change_steps)

            # select an action
            action, _ = select_action(loaded_policy, observation)

            # save data for the step
            actions.append(action)
            observations.append(observation)

            # take a step in the env, check termination, etc.
            observation, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
                if reward > 0:
                    nb_rewards += 1

        # stack and save data for the trajectory
        observations = np.stack(observations)
        actions = np.stack(actions)

        trajectories.append(observations)
        trajectories_action.append(actions)

    print("proportion of rewarded trials :", nb_rewards / nb_trajectories)

    # stack the trajectories and convert to grayscale
    trajectories = np.stack(trajectories)
    trajectories_action = np.stack(trajectories_action)
    trajectories = np.sum(trajectories, axis=-3, keepdims=1) / 3

    env.close()

    # save the (global) data with all trajectory rollouts collected
    with open(f"../data/rollout_{collect_type}_obs_{id_run}.npy", "wb") as f:
        np.save(f, trajectories)
    with open(f"../data/rollout_{collect_type}_actions_{id_run}.npy", "wb") as f:
        np.save(f, trajectories_action)

    del trajectories, trajectories_action, env
    torch.cuda.empty_cache()


def collect_rollouts():
    i_run = 4
    collect_type = "constant_gain"
    run_trajectories(nb_trajectories=10, id_run=i_run, collect_type=collect_type)

    collect_type = "changing_gain_straight"
    run_trajectories(nb_trajectories=10, id_run=i_run, collect_type=collect_type)

    collect_type = "changing_gain_straight_glitch"
    run_trajectories(nb_trajectories=20, id_run=i_run, collect_type=collect_type)
