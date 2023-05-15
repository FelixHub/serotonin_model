"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import argparse
import math
import gymnasium as gym
import pyglet
from pyglet.window import key
import miniworld
import numpy as np
import yaml

with open("default_parameters.yaml") as file:
    default_parameters = yaml.load(file, Loader=yaml.FullLoader)

env_args= dict(
        min_section_length=default_parameters['min_section_length'],
        max_section_length=default_parameters['max_section_length'],
        max_episode_steps=default_parameters['max_episode_steps'],
        facing_forward=default_parameters['facing_forward'],
        reset_keep_same_length=default_parameters['reset_keep_same_length'],
        wall_tex="stripe_gradient" # default_parameters['wall_tex'],
            )


parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default='MiniWorld-TaskHallwaySimple-v0') # miniworld.envs.env_ids[0] MiniWorld-OneRoomS6-v0
parser.add_argument(
    "--domain-rand", action="store_true", help="enable domain randomization"
)
parser.add_argument(
    "--no-time-limit", action="store_true", help="ignore time step limits"
)
parser.add_argument(
    "--top_view",
    action="store_true",
    help="show the top view instead of the agent view",
)
args = parser.parse_args()
view_mode = "top" if args.top_view else "agent"

env = gym.make(args.env_name, view=view_mode, render_mode="human",**env_args)

if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True

print("============")
print("Instructions")
print("============")
print("move: arrow keys\npickup: P\ndrop: D\ndone: ENTER\nquit: ESC")
print("============")

env.reset()

# Create the display window
env.render()


def step(action):
    print(
        "step {}/{}: {}".format(env.step_count + 1, env.max_episode_steps, env.actions(action).name))

    obs, reward, termination, truncation, info = env.step(action)

    if reward > 0:
        print(f"reward={reward:.2f}")

    if env.step_count % 5 == 0 :
        env.change_gain(random=True,motor_gains=[1, 2, 4],glitch=False,glitch_phase=np.random.uniform(-0.8,0.8)) # glitch from -0.8 to +0.8 with 0.2 intervals

    if termination or truncation:
        print("done!")
        env.reset()

    env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
        return

    if symbol == key.ESCAPE:
        env.close()
        # sys.exit(0)

    if symbol == key.UP:
        step(env.actions.move_forward)
    elif symbol == key.DOWN:
        step(env.actions.move_back)

    elif symbol == key.LEFT:
        step(env.actions.turn_left)
    elif symbol == key.RIGHT:
        step(env.actions.turn_right)

    elif symbol == key.PAGEUP or symbol == key.P:
        step(env.actions.pickup)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(env.actions.drop)

    elif symbol == key.ENTER:
        step(env.actions.done)


@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass


@env.unwrapped.window.event
def on_draw():
    env.render()


@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()


# Enter main event loop
pyglet.app.run()

env.close()