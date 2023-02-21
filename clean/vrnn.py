import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# we want to use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def kl_dist(logvar1,mean1,logvar2,mean2):
    return -0.5 * torch.sum(logvar1 - logvar2 - torch.div((logvar1.exp() + (mean1 - mean2).pow(2)), logvar2.exp()+1e-10))

class VRNN(nn.Module):
    '''
    The Varational Recurrent Neural Network VRNN is a dynamical VAE model.
    This is a re-writing from "Dynamical Variational Autoencoders: A Comprehensive Review" https://arxiv.org/abs/2008.12595
    '''

    def __init__(self, img_size=64, hidden_dim=128, latent_dim=32, RNN_dim=32, CNN_channels=64):

        super(VRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rnn_dim = RNN_dim

        # convolutions parameters
        self.CNN_channels = CNN_channels
        self.img_size = img_size
        self.feature_size_enc = 14*14*CNN_channels # output of convolutional layers
        self.feature_size_dec = 12*12*CNN_channels # inpout of transposed convolutional layers
        self.feature_size = (img_size-8)*(img_size-8)*CNN_channels # output of convolutional layers

        # feature extractor of observation with convolutional layers
        self.enc_conv1 = nn.Conv2d(1, CNN_channels//2, kernel_size=5)
        self.enc_conv2 = nn.Conv2d(CNN_channels//2, CNN_channels, kernel_size=5)
        self.enc_maxpool = nn.MaxPool2d(4)

        # encoder part of the model
        self.encoder = nn.Sequential(
            nn.Linear( self.feature_size_enc + RNN_dim, hidden_dim),
            nn.Tanh()
        )
        self.enc_mean = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # encoding the prior on the latent space given previous dynamics
        self.prior_encoder = nn.Sequential(
            nn.Linear(RNN_dim, hidden_dim),
            nn.Tanh()
        )
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        # feature extractor of latent states with inverse convolutional layers
        self.dec_conv1 = nn.ConvTranspose2d(CNN_channels, CNN_channels//2, kernel_size=5, stride=5)
        self.dec_conv2 = nn.ConvTranspose2d(CNN_channels//2, 1, kernel_size=5)

        # decoder part of the model, output the reconstructed observation y_t
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + RNN_dim, hidden_dim),
            nn.Tanh()
        )
        self.dec_mean = nn.Linear(hidden_dim, self.feature_size_dec)
        # self.dec_logvar = nn.Linear(hidden_dim, self.feature_size)

        # RNN part of the model, encode the dynamics of the latent space
        self.rnn = nn.RNN(self.feature_size_enc + latent_dim, RNN_dim)#

        self.relu = nn.ReLU()

    def reparametrization(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def forward(self,x,generation_mode=False,generation_ratio=0):

        seq_len, batch_size,_,_ = x.shape

        # create variable holder
        self.z_mean = torch.zeros(seq_len, batch_size, self.latent_dim).to(device)
        self.z_logvar = torch.zeros(seq_len, batch_size, self.latent_dim).to(device)
        self.z = torch.zeros(seq_len, batch_size, self.latent_dim).to(device)

        h = torch.zeros((seq_len, batch_size, self.rnn_dim)).to(device)
        z_t = torch.zeros((batch_size, self.latent_dim)).to(device)
        h_t = torch.zeros((batch_size, self.rnn_dim)).to(device) # the initial hidden state coudl be different from 0 ... (random ?)

        y = torch.zeros((seq_len, batch_size, self.img_size,self.img_size)).to(device) # observation reconstruction

        for t in range(seq_len):

            x_t = x[t,:,:,:].unsqueeze(1)

            # feature extractor of observation with convolutional layers
            phi_x_t = self.relu(self.enc_conv1(x_t))
            phi_x_t = self.enc_conv2(phi_x_t)
            phi_x_t = self.enc_maxpool(phi_x_t)
            phi_x_t = self.relu(phi_x_t)
            phi_x_t = phi_x_t.view(-1, self.feature_size_enc )
            encoder_input = torch.cat((phi_x_t,h_t),dim=1)
            encoder_output = self.encoder(encoder_input)
            mean_zt, logvar_zt = self.enc_mean(encoder_output), self.enc_logvar(encoder_output)

            # sample the latent variable
            if generation_mode and t > 0 :
                prior_encoder_t = self.prior_encoder(h_t)
                prior_mean_zt = self.prior_mean(prior_encoder_t)
                prior_logvar_zt = self.prior_logvar(prior_encoder_t)
                z_t = self.reparametrization(prior_mean_zt, prior_logvar_zt)
            else :
                z_t = self.reparametrization(mean_zt, logvar_zt)

            # decode
            decoder_input = torch.cat((z_t,h_t),dim=1)
            decoder_output = self.decoder(decoder_input)
            mean_phi_z_t = self.dec_mean(decoder_output) # for now we will not sample from the decoder

            # reconstruction of the observation with convolutional layers
            phi_z_t = mean_phi_z_t.view(-1, self.CNN_channels, 12, 12)
            #print('before deconv',phi_z_t.shape)
            y_t = self.relu(self.dec_conv1(phi_z_t))
            #print('after first deconv',y_t.shape)
            y_t = torch.sigmoid(self.dec_conv2(y_t)).squeeze(1)
            # print('after second deconv',y_t.shape)

            # update the hidden state
            if generation_mode :
                random_gen = np.random.uniform(size=1)[0] < generation_ratio
            else :
                random_gen = False

            if random_gen :
                phi_x_t_hat = self.relu(self.enc_conv1(y_t.unsqueeze(1)))
                phi_x_t_hat = self.enc_conv2(phi_x_t_hat)
                phi_x_t_hat = self.enc_maxpool(phi_x_t_hat)
                phi_x_t_hat = self.relu(phi_x_t_hat)
                phi_x_t_hat = phi_x_t_hat.view(-1, self.feature_size_enc )
                rnn_input = torch.cat((phi_x_t_hat,z_t),dim=1)
            else :
                rnn_input = torch.cat((phi_x_t,z_t),dim=1)

            h_t = self.rnn(rnn_input.unsqueeze(0),h_t.unsqueeze(0))[1].squeeze(0)

            # save variable
            y[t,:,:,:] = y_t
            self.z_mean[t,:,:] = mean_zt
            self.z_logvar[t,:,:] = logvar_zt
            self.z[t,:,:] = z_t
            h[t,:,:] = h_t

        # generation of the latent variable z prior (for the KL divergence)
        prior_encoder_output = self.prior_encoder(h)
        self.z_prior_mean = self.prior_mean(prior_encoder_output)
        self.z_prior_logvar = self.prior_logvar(prior_encoder_output)

        return y,  self.z_mean, self.z_logvar, self.z_prior_mean, self.z_prior_logvar, self.z


    def loss_function(self,x_reconstructed, x, mean, logvar, mean_prior=None, logvar_prior=None):
        if mean_prior is None :
            mean_prior = torch.zeros_like(mean)
        if logvar_prior is None :
            logvar_prior = torch.zeros_like(logvar)
        # reconstruction loss
        recon_loss = F.binary_cross_entropy(x_reconstructed, x,reduction='sum')
        # KL divergence
        kl_loss = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mean - mean_prior).pow(2)), logvar_prior.exp()+1e-10))
        return recon_loss + kl_loss


