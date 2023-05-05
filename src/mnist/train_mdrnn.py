import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.mdrnn import MDRNN
from models.vae_64_64 import VAE_64_64

# checking if the saved_models directory exists or create it
if not os.path.isdir("../saved_models/MNIST"):
    os.makedirs("../saved_models/MNIST")

# we want to use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device used:", device)

# import data from the moving mnist dataset
data = np.load("../data/MNIST/bouncing_mnist_test.npy")
data = data / 255  # normalize the data to be between 0 and 1
seq_len = data.shape[1] 

def count_parameters(net):
    # return the number of parameters of the model
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def main(params_path, model_path, model_vae_path):
    # we do not allow the training if there is not a training VAE model
    if model_vae_path is None:
        raise ValueError("no VAE model path given, please train a VAE model first.")
    else:
        print("loading VAE model saved in", model_vae_path)

        vae_params_path_recovery = model_vae_path[:-3] + ".params.yml"
        with open(vae_params_path_recovery) as file:
            vae_parameters = yaml.load(file, Loader=yaml.FullLoader)
        print("using VAE parameters from " + vae_params_path_recovery)

        vae_model = VAE_64_64(
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

    # even if we load a model, we are going to save a new version of it to not risk any overwriting
    model_name = "model_mdrnn_" + datetime.today().strftime("%Y-%m-%d")

    # we save the corresponding parameters
    with open(
        "../saved_models/MNIST/" + model_name + ".params.yml", "w"
    ) as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)

    model = model.to(device)
    batch_size = parameters["batch_size_mdrnn"]
    train_loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=parameters["learning_rate_mdrnn"]
    )
    print("number of parameters :", count_parameters(model))

    nb_epochs = parameters["nb_epochs_mdrnn"]
    for epoch in range(nb_epochs):
        pbar = tqdm(train_loader)
        pbar.set_description("epoch %s/ loss=? " % str(epoch + 1))
        for x in pbar:
            x = torch.reshape(x, (-1, 1, 64, 64)).float().to(device)
            latents = vae_model.encoder(x)[0]
            latents = torch.transpose(torch.reshape(latents, (-1, seq_len, model.latent_dim)), 0, 1)
            _, batch_size, _ = latents.shape

            if model.memory == "rnn":
                hidden = torch.zeros(batch_size, model.hidden_dim).to(device)
            else:
                hidden = (
                    torch.zeros(batch_size, model.hidden_dim).to(device),
                    torch.zeros(batch_size, model.hidden_dim).to(device),
                )
            action = torch.zeros(batch_size, model.action_dim).to(device)

            optimizer.zero_grad()
            episode_loss = []
            for t in range(seq_len - 1):
                latent = latents[t, :, :]
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
                model.state_dict(), "../saved_models/MNIST/" + model_name + ".pt"
            )

    torch.save(model.state_dict(), "../saved_models/MNIST/" + model_name + ".pt")
    print(
        "end of training, model saved at",
        "saved_models/MNIST/" + model_name + ".pt",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line options")
    parser.add_argument(
        "--params_path", type=str, dest="params_path", default="mnist/default_parameters.yaml"
    )
    parser.add_argument("--model_path", type=str, dest="model_path", default=None)
    possible_vae_path = "../saved_models/MNIST/model_vae_" + datetime.today().strftime("%Y-%m-%d") + ".pt"
    parser.add_argument(
        "--model_vae_path", type=str, dest="model_vae_path", default=possible_vae_path
    )
    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items()})
