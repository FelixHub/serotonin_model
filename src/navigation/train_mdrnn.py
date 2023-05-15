import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.mdrnn import MDRNN
from models.vae_60_80 import VAE_60_80
from torch.utils.data import Dataset

# checking if the saved_models directory exists or create it
if not os.path.isdir("../saved_models/navigation"):
    os.makedirs("../saved_models/navigation")

# we want to use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device used:", device)


def count_parameters(net):
    # return the number of parameters of the model
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

class trajectoryDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB 
    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        return xA, xB
    def __len__(self):
        return len(self.datasetA)

def get_navigation_training_data(parameters):

    data = []
    data_actions = []

    rollout_type = parameters['rollout_type_mdrnn']
    print('using training data from',rollout_type)
    nb_rollout_files = 2 # len(os.listdir('../data/navigation/')) // 6

    for i in range(0,nb_rollout_files):
        data_temp = np.load('../data/navigation/'+rollout_type+'_obs_'+str(i)+'_alt_texture_1.npy')
        data_temp = np.transpose(data_temp,(0,1,2,4,3))
        data.append(data_temp)

        data_actions_temp = np.load('../data/navigation/'+rollout_type+'_actions_'+str(i)+'_alt_texture_1.npy')
        data_actions.append(data_actions_temp)

    data = np.concatenate(data) / 255
    print("shape observations",data.shape)

    data_actions = np.concatenate(data_actions)
    data_actions = torch.tensor(data_actions)
    data_actions = torch.nn.functional.one_hot(data_actions.long())
    print("shape actions",data_actions.shape)

    nb_trajectories = data_actions.shape[0]
    nb_steps = data_actions.shape[1]

    trajectory_dataset = trajectoryDataset(data, data_actions)
    train_loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=parameters['batch_size_mdrnn'], shuffle=True)
    joint_loader = torch.utils.data.DataLoader(
            dataset=trajectory_dataset, batch_size=parameters['batch_size_mdrnn'], shuffle=True
        )
    return train_loader, joint_loader, nb_trajectories, nb_steps

def main(params_path, model_path, model_vae_path,save_each_epochs,new_rollout_mode):
    # we do not allow the training if there is not a training VAE model
    if model_vae_path is None:
        raise ValueError("no VAE model path given, please train a VAE model first.")
    else:
        print("loading VAE model saved in", model_vae_path)

        vae_params_path_recovery = model_vae_path[:-3] + ".params.yml"
        with open(vae_params_path_recovery) as file:
            vae_parameters = yaml.load(file, Loader=yaml.FullLoader)
        print("using VAE parameters from " + vae_params_path_recovery)

        vae_model = VAE_60_80(
            img_channels=vae_parameters["img_channels_vae"],
            latent_dim=vae_parameters["latent_dim_vae"],
            beta=vae_parameters["beta_vae"],
        )
        vae_model.load_state_dict(torch.load(model_vae_path))
        vae_model.to(device)
        print("number of parameters in vae :", count_parameters(vae_model))

    # now we create the MDRNN model
    if model_path is None:
        print("no model path given, training from scratch using default parameters.")
        with open(params_path) as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)
        model = MDRNN(
            latent_dim=parameters["latent_dim_mdrnn"],
            action_dim=parameters["action_dim_mdrnn"],
            hidden_dim=parameters["hidden_dim_mdrnn"],
            gaussians_nb=parameters["gaussians_nb_mdrnn"],
        )
    else:
        print("loading model saved in", model_path)
        try:
            params_path_recovery = model_path[:-3] + ".params.yml"
            with open(params_path_recovery) as file:
                parameters = yaml.load(file, Loader=yaml.FullLoader)
            print("using parameters from " + params_path_recovery)
        except:
            print("model parameters not found, trying to use default parameters.")
            with open(params_path) as file:
                parameters = yaml.load(file, Loader=yaml.FullLoader)
        model = MDRNN(
            latent_dim=parameters["latent_dim_mdrnn"],
            action_dim=parameters["action_dim_mdrnn"],
            hidden_dim=parameters["hidden_dim_mdrnn"],
            gaussians_nb=parameters["gaussians_nb_mdrnn"],
        )
        model.load_state_dict(torch.load(model_path))


    # if we want to train the model with a different rollout that its parameters, we overide the parameters
    if new_rollout_mode is not None :
        model_name = "model_mdrnn_"+ parameters['rollout_type_mdrnn'] +"_to_"+new_rollout_mode +'_'+datetime.today().strftime("%Y-%m-%d")
        parameters['rollout_type_mdrnn'] = new_rollout_mode
    else :
        # even if we load a model, we are going to save a new version of it to not risk any overwriting
        model_name = "model_mdrnn_1_"+ parameters['rollout_type_mdrnn'] +'_'+datetime.today().strftime("%Y-%m-%d")

    # we save the corresponding parameters
    with open(
        "../saved_models/navigation/" + model_name + ".params.yml", "w"
    ) as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=parameters["learning_rate_mdrnn"]
    )

    train_loader, joint_loader, nb_trajectories, nb_steps = get_navigation_training_data(parameters)
    
    print("number of parameters :", count_parameters(model))

    if save_each_epochs :
        torch.save(
            model.state_dict(), "../saved_models/navigation/" + model_name + "_epoch_"+str(0)+".pt"
        )

    nb_epochs = parameters["nb_epochs_mdrnn"]
    for epoch in range(nb_epochs):
        pbar = tqdm(joint_loader)
        pbar.set_description("epoch %s/ loss=? " % str(epoch + 1))
        for obs,act in pbar:

            x = torch.reshape(obs, (-1, 1, 60, 80)).float().to(device)
            latents = vae_model.encoder(x)[0]
            latents = torch.transpose(torch.reshape(latents, (-1, nb_steps, parameters['latent_dim_vae'])), 0, 1)
            
            seq_len, batch_size, _ = latents.shape

            if model.memory == "rnn":
                hidden = torch.zeros(batch_size, model.hidden_dim).to(device)
            else:
                hidden = (
                    torch.zeros(1,batch_size, model.hidden_dim).to(device),
                    torch.zeros(1,batch_size, model.hidden_dim).to(device),
                )
            actions = torch.transpose(act,0,1).to(device)
            optimizer.zero_grad()
            episode_loss = []
            
            for t in range(seq_len - 1):
                latent = latents[t, :, :]
                action = actions[t,:,:]
                latent_next_obs = latents[t + 1, :, :]

                if model.memory == "rnn":
                    mus, sigmas, logpi, next_hidden = model(action, latent, hidden)
                    hidden = next_hidden[1].squeeze(0)
                else:
                    mus, sigmas, logpi, next_hidden = model(action, latent, hidden)
                    hidden = next_hidden

                loss = model.loss_function(mus, sigmas, logpi, latent_next_obs)
                episode_loss.append(loss)

            episode_loss = torch.stack(episode_loss).sum()
            episode_loss.backward()
            optimizer.step()

            pbar.set_description(
                "epoch "
                + str(epoch + 1)
                + "/"
                + str(nb_epochs)
                + "  loss= "
                + str(loss.cpu().detach().numpy())
            )
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(), "../saved_models/navigation/" + model_name + ".pt"
            )
        if save_each_epochs :
            torch.save(
                model.state_dict(), "../saved_models/navigation/" + model_name + "epoch_"+str(epoch + 1)+".pt"
            )

    torch.save(model.state_dict(), "../saved_models/navigation/" + model_name + ".pt")
    print(
        "end of training, model saved at",
        "saved_models/navigation/" + model_name + ".pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line options")
    parser.add_argument(
        "--params_path", type=str, dest="params_path", default="navigation/default_parameters.yaml"
    )
    parser.add_argument("--model_path", type=str, dest="model_path", default=None)
    possible_vae_path = "../saved_models/navigation/model_vae_" + datetime.today().strftime("%Y-%m-%d") + ".pt"
    parser.add_argument(
        "--model_vae_path", type=str, dest="model_vae_path", default=possible_vae_path
    )
    parser.add_argument(
        "--save_each_epochs", type=str, dest="save_each_epochs", default=False
    )
    parser.add_argument(
        "--new_rollout_mode", type=str, dest="new_rollout_mode", default=None
    )
    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items()})


'''

main(
    params_path="navigation/default_parameters.yaml",
    model_path="../saved_models/navigation/model_mdrnn_1_rollout_changing_gain_straight_2023-05-08.pt",
    model_vae_path="../saved_models/navigation/model_vae_2023-05-07.pt",
    save_each_epochs=False,
    new_rollout_mode=None
)

'''