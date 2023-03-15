import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.vrnn import VRNN_OLD

# checking if the saved_models directory exists or create it
if not os.path.isdir("../saved_models"):
    os.makedirs("../saved_models")

# we want to use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device used:", device)

# import data from the moving mnist dataset
data = np.load("../data/bouncing_mnist_test.npy")
data = data / 255  # normalize the data to be between 0 and 1


def count_parameters(net):
    # return the number of parameters of the model
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def main(params_path, model_path):
    if model_path is None:
        print("no model path given, training from scratch using default parameters.")
        with open(params_path) as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)
        model = VRNN_OLD(
            img_size=parameters["img_size"],
            hidden_dim=parameters["hidden_dim"],
            latent_dim=parameters["latent_dim"],
            RNN_dim=parameters["RNN_dim"],
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
        model = VRNN_OLD(
            img_size=parameters["img_size"],
            hidden_dim=parameters["hidden_dim"],
            latent_dim=parameters["latent_dim"],
            RNN_dim=parameters["RNN_dim"],
        )
        model.load_state_dict(torch.load(model_path))

    # even if we load a model, we are going to save a new version of it to not risk any overwriting
    model_name = "model_vrnn_" + datetime.today().strftime("%Y-%m-%d")

    # we save the corresponding parameters
    with open("../saved_models/" + model_name + ".params.yml", "w") as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)

    model = model.to(device)
    batch_size = parameters["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    print("number of parameters :", count_parameters(model))

    nb_epochs = parameters["nb_epochs"]
    for epoch in range(nb_epochs):
        pbar = tqdm(train_loader)
        pbar.set_description("epoch %s/ loss=? " % str(epoch + 1))
        for x in pbar:
            x = torch.reshape(x, (20, -1, 64, 64)).float().to(device)
            optimizer.zero_grad()
            y, mean, logvar, mean_prior, logvar_prior, z = model.forward(x)
            loss = model.loss_function(y, x, mean, logvar, mean_prior, logvar_prior)
            loss.backward()
            optimizer.step()
            pbar.set_description(
                "epoch "
                + str(epoch + 1)
                + "/"
                + str(nb_epochs)
                + "  loss= "
                + str(loss.cpu().detach().numpy())
            )
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), "../saved_models/" + model_name + ".pt")
    torch.save(model.state_dict(), "../saved_models/" + model_name + ".pt")
    print("end of training, model saved at", "saved_models/" + model_name + ".pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line options")
    parser.add_argument(
        "--params_path", type=str, dest="params_path", default="default_parameters.yaml"
    )
    parser.add_argument("--model_path", type=str, dest="model_path", default=None)
    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items()})
