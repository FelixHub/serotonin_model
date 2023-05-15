import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.vae_60_80 import VAE_60_80

# checking if the saved_models directory exists or create it
if not os.path.isdir("../saved_models/navigation"):
    os.makedirs("../saved_models/navigation")

# we want to use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device used:", device)

# import random rollout data from the Task-Hallway environment
data = np.load("../data/navigation/randomRollout_alt_texture.npy")
data = data / 255
print("random rollout data shape :",data.shape)

def count_parameters(net):
    # return the number of parameters of the model
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def main(params_path, model_path):
    if model_path is None:
        print("no model path given, training from scratch using default parameters.")
        with open(params_path) as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)
        model = VAE_60_80(
            img_channels=parameters["img_channels_vae"],
            latent_dim=parameters["latent_dim_vae"],
            beta=parameters["beta_vae"],
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
        model = VAE_60_80(
            img_channels=parameters["img_channels_vae"],
            latent_dim=parameters["latent_dim_vae"],
            beta=parameters["beta_vae"],
        )
        model.load_state_dict(torch.load(model_path))

    # even if we load a model, we are going to save a new version of it to not risk any overwriting
    model_name = "model_vae_" + datetime.today().strftime("%Y-%m-%d")

    # we save the corresponding parameters
    with open(
        "../saved_models/navigation/" + model_name + ".params.yml", "w"
    ) as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)

    model = model.to(device)
    batch_size = parameters["batch_size_vae"]
    train_loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate_vae"])
    print("number of parameters :", count_parameters(model))

    nb_epochs = parameters["nb_epochs_vae"]
    for epoch in range(nb_epochs):
        pbar = tqdm(train_loader)
        pbar.set_description("epoch %s/ loss=? " % str(epoch + 1))

        for x in pbar:
            imgs = x
            imgs = imgs.to(device).float()
            imgs = imgs.unsqueeze(1).squeeze(-1)
            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logsigma = model(imgs)

            loss, loss_KL = model.loss_function(out, imgs, mu, logsigma)

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                "epoch "
                + str(epoch + 1)
                + "/"
                + str(nb_epochs)
                + " loss= "
                + str(loss.cpu().detach().numpy())
                + " loss KL = "
                + str(loss_KL.cpu().detach().numpy())
            )

        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(), "../saved_models/navigation/" + model_name + ".pt"
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
    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items()})
