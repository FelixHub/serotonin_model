# Serotonin and state prediction errors

This is the project repository for the NeurIPS 2023 article on serotonin and state prediction errors.

## Installation

We recommend using [mamba](https://github.com/mamba-org/vscode-micromamba) as the reference manager. It is way faster than conda and basically a drop-in replacement (replace `conda` with `mamba`). Installation is fairly simple:

```
# clone this repository and change directory the repository directory
git clone https://github.com/FelixHub/serotonin_model.git
cd serotonin_model/

# create and activate environment (use conda instead of mamba if you like)
mamba create --name serotonin --file requirements.txt
mamba activate serotonin
```

## Usage

To avoid path issues, make sure to change your directory to `src` by `cd src`.

### MNIST Data Generation

All modules are present in `src/mnist` with an example data generation notebook in `src/view_generated_data.ipynb`.

### Training the world models

Training scripts are present in `src` -> `src/train_mdrnn.py`, `src/train_vae.py`, and `src/train_vrnn.py`.

#### VAE-MDRNN (Ha & Schmidhuber)

To train the VAE-MDRNN model, you need to first train the VAE model, and then to train the MDRNN model by specifying the path of the trained VAE model.

- edit `src/default_parameters.yaml` with the desired parameters for VAE & MDRNN and their trainings
- run `src/train_vae.py` with no argument. It will create a folder `saved_models` where the models and their parameter files will be saved. 
- run `train_mdrnn --vae_model_path="path_of_your_trained_vae_model"`.

You can re-train the models by specifying `--model_path`

To analyse the VAE-MDRNN model, use the jupyter notebook `analysis.ipynb`.

#### VRNN

Use the same steps as above but with `src/train_vrnn.py`.