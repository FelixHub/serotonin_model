import numpy as np
import torch 
from tqdm import tqdm
import yaml
from models.vrnn import VRNN
import sys
import argparse
from datetime import datetime
import os
  
# checking if the saved_models directory exists or create it
if not os.path.isdir("saved_models"):
    os.makedirs("saved_models")

# we want to use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device used:",device)

# import data from the moving mnist dataset
data = np.load('data/bouncing_mnist_test.npy')
data = data / 255 # normalize the data to be between 0 and 1

def count_parameters(net):
    # return the number of parameters of the model
    return sum(p.numel() for p in net.parameters() if p.requires_grad)