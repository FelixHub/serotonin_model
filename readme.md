## clean code for training and analyzing VRNN models on the moving MNIST dataset

#### Training the VAE-MDRNN model

To train the VAE-MDRNN model, you need to first train the VAE model, and then to train the MDRNN model by specifying the path of the trained VAE model.
- edit default_parameters.yaml with the desired parameters for VAE & MDRNN and their trainings.
- run train_vae.py with no argument. It will create a folder "saved_models" where the models and their parameter files will be saved. 
- run train_mdrnn --vae_model_path="path_of_your_trained_vae_model" .
You can re-train the models by specifying --model_path .

To analyse the VAE-MDRNN model, use the jupyter notebook analysis.ipynb. 