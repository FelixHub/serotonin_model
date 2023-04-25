from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import torch
from miniworld.wrappers import PyTorchObsWrapper
from tqdm import tqdm

from .agent import DEVICE, Policy, select_action

loaded_policy = Policy().to(DEVICE)
loaded_policy.load_state_dict(torch.load("../saved_models/miniworld_task.pt"))
loaded_policy.eval()


class BaseCollector(ABC):
    def __init__(self, env, policy, n_rollouts, id_run):
        self.env = env
        self.policy = policy
        self.n_rollouts = n_rollouts
        self.id_run = id_run
        self.args = {}

    @abstractmethod
    def strategy(self, i_step, gain_change_steps):
        pass

    def collect(self):
        rollouts = []
        rollouts_action = []
        n_rewards = 0

        for i_rollouts in tqdm(range(self.n_rollouts)):
            if i_rollouts % 50 == 0:
                self.env = PyTorchObsWrapper(self.env)

            observations = []
            actions = []
            observation, _ = self.env.reset()
            gain_change_steps = self.initialize_gain_change_steps()

            for i_step in range(self.args["max_episode_steps"]):
                self.strategy(i_step, gain_change_steps)

                action, _ = select_action(self.policy, observation)
                actions.append(action)
                observations.append(observation)

                observation, reward, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    observation, _ = self.env.reset()
                    if reward > 0:
                        n_rewards += 1

            observations = np.stack(observations)
            actions = np.stack(actions)

            rollouts.append(observations)
            rollouts_action.append(actions)

        return rollouts, rollouts_action, n_rewards

    def initialize_gain_change_steps(self):
        return []


class ConstantGainCollector(BaseCollector):
    def __init__(self, env, policy, n_rollouts, id_run):
        super().__init__(env, policy, n_rollouts, id_run)
        self.args = dict(
            n_sections=3,
            proba_change_motor_gain=0,
            min_section_length=3,
            max_section_length=8,
            training=False,
            max_episode_steps=100,
        )

    def strategy(self, i_step, gain_change_steps):
        pass


class ChangingGainStraightCollector(BaseCollector):
    def __init__(self, env, policy, n_rollouts, id_run):
        super().__init__(env, policy, n_rollouts, id_run)
        self.args = dict(
            n_sections=1,
            proba_change_motor_gain=0,
            min_section_length=25,
            max_section_length=40,
            training=True,
            max_episode_steps=100,
        )

    def strategy(self, i_step, gain_change_steps):
        if (len(gain_change_steps) > 0) and (i_step == gain_change_steps[0]):
            self.env.change_gain(motor_gains=[0.5, 1, 1.5])
            gain_change_steps.pop(0)

    def initialize_gain_change_steps(self):
        n_changes = np.random.randint(1, 5)
        return list(
            np.sort(np.random.choice(range(1, 10), size=n_changes, replace=False) * 10)
        )


class ChangingGainStraightGlitchCollector(ChangingGainStraightCollector):
    def strategy(self, i_step, gain_change_steps):
        if (len(gain_change_steps) > 0) and (i_step == gain_change_steps[0]):
            self.env.change_gain(
                motor_gains=[0.5, 1, 1.5],
                glitch=True,
                glitch_phase=np.random.choice(
                    [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
                ),
            )
            gain_change_steps.pop(0)


def collect_rollouts():
    i_run = 4
    rollout_params = [
        ("constant_gain", ConstantGainCollector, 10),
        ("changing_gain_straight", ChangingGainStraightCollector, 10),
        ("changing_gain_straight_glitch", ChangingGainStraightGlitchCollector, 10),
    ]

    for collect_type, collector_class, n_rollouts in rollout_params:
        env = gym.make("MiniWorld-TaskHallway-v0", view="agent", render_mode=None)
        collector = collector_class(
            env, loaded_policy, n_rollouts=n_rollouts, id_run=i_run
        )
        rollouts, rollouts_action, n_rewards = collector.collect()
        save_rollouts(rollouts, rollouts_action, n_rewards, collect_type, i_run)


def save_rollouts(rollouts, rollouts_action, n_rewards, collect_type, i_run):
    print("proportion of rewarded trials :", n_rewards / len(rollouts))

    rollouts = np.stack(rollouts)
    rollouts_action = np.stack(rollouts_action)
    rollouts = np.sum(rollouts, axis=-3, keepdims=1) / 3

    with open(f"../data/rollout_{collect_type}_obs_{i_run}.npy", "wb") as f:
        np.save(f, rollouts)
    with open(f"../data/rollout_{collect_type}_actions_{i_run}.npy", "wb") as f:
        np.save(f, rollouts_action)

    del rollouts, rollouts_action
    torch.cuda.empty_cache()


if __name__ == "__main__":
    collect_rollouts()
