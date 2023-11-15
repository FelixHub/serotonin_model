from navigation.data_generation.rollouts import collect_rollouts
import yaml

i_run=1

nb_trajectories = 3000
params_path = "navigation/default_parameters.yaml"

with open(params_path) as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)

env_args= dict(
                min_section_length=parameters['min_section_length'],
                max_section_length=parameters['max_section_length'],
                max_episode_steps=parameters['max_episode_steps'],
                facing_forward=parameters['facing_forward'],
                reset_keep_same_length=parameters['reset_keep_same_length'],
                wall_tex=parameters['wall_tex'],
                )

collect_rollouts(i_run,nb_trajectories,env_args)