import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

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

        self.img_size=img_size

        ### POSTERIOR ENCODER ###
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2) # output of size 1024

        self.fc_x_enc = nn.Linear( 1024 + RNN_dim, hidden_dim)
        self.enc_mean = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        ### PRIOR ENCODER ###
        self.fc_z_prior = nn.Linear(RNN_dim, hidden_dim)
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        ### DECODER ###
        self.fc_z_dec_1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_z_dec_2 = nn.Linear(hidden_dim + RNN_dim, hidden_dim)
        self.fc_z_dec_3 = nn.Linear(hidden_dim, 1024)

        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 6, stride=2)

        #### MEMORY NETWORK ###
        self.rnn = nn.RNN(hidden_dim+1024, RNN_dim)

        self.relu = nn.ReLU()

    def reparametrization(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def forward(self,x):

        seq_len, batch_size,_,_ = x.shape

        # variables holders
        self.z_mean = torch.zeros(seq_len, batch_size, self.latent_dim).to(device)
        self.z_logvar = torch.zeros(seq_len, batch_size, self.latent_dim).to(device)
        self.z = torch.zeros(seq_len, batch_size, self.latent_dim).to(device)

        h = torch.zeros((seq_len, batch_size, self.rnn_dim)).to(device)
        z_t = torch.zeros((batch_size, self.latent_dim)).to(device)
        h_t = torch.zeros((batch_size, self.rnn_dim)).to(device)

        y = torch.zeros((seq_len, batch_size, self.img_size,self.img_size)).to(device) # observation reconstruction

        for t in range(seq_len):

            x_t = x[t,:,:,:].unsqueeze(1)

            # ENCODING
            x_t = F.relu(self.conv1(x_t))
            x_t = F.relu(self.conv2(x_t))
            x_t = F.relu(self.conv3(x_t))
            x_t = F.relu(self.conv4(x_t))
            phi_x_t = x_t.view(x_t.size(0), -1)
            encoding = torch.cat((phi_x_t,h_t),dim=1) # concatenate the hidden state with the transformed image
            encoding = F.relu(self.fc_x_enc(encoding))
            mean_zt = self.enc_mean(encoding)
            logvar_zt = self.enc_logvar(encoding) 
            z_t = self.reparametrization(mean_zt,logvar_zt)

            # DECODING
            phi_z_t = F.relu(self.fc_z_dec_1(z_t))
            decoding = torch.cat((phi_z_t,h_t),dim=1) # concatenate the hidden state with the transformed latent variable
            decoding = F.relu(self.fc_z_dec_2(decoding))
            decoding = F.relu(self.fc_z_dec_3(decoding))
            x_rec = decoding.unsqueeze(-1).unsqueeze(-1)
            x_rec = F.relu(self.deconv1(x_rec))
            x_rec = F.relu(self.deconv2(x_rec))
            x_rec = F.relu(self.deconv3(x_rec))
            y_t = torch.sigmoid(self.deconv4(x_rec)).squeeze(1)

            # RNN UPDATE
            rnn_input = torch.cat((phi_x_t,phi_z_t),dim=1)
            h_t = self.rnn(rnn_input.unsqueeze(0),h_t.unsqueeze(0))[1].squeeze(0)

            # save variables
            y[t,:,:,:] = y_t
            self.z_mean[t,:,:] = mean_zt
            self.z_logvar[t,:,:] = logvar_zt
            self.z[t,:,:] = z_t
            h[t,:,:] = h_t

        # PRIOR ENCODER
        prior_encoder_output = F.relu(self.fc_z_prior(h))
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


class VRNN_OLD(nn.Module):

    '''
    the Varational Recurrent Neural Network VRNN is a dynamical VAE model.
    This is the first version which for some reason work way better than the new one...
    '''

    def __init__(self, img_size=64, hidden_dim=256, latent_dim=16, RNN_dim=256, CNN_channels=32):

        super(VRNN_OLD, self).__init__()

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

    def encoder_pass(self,x):
        '''Here we want to look at the latent space of the VRNN, so we get as input only one image of size (1,1,64,64)'''
        print('x.shape',x.shape)
        y,  z_mean, z_logvar, z_prior_mean, z_prior_logvar, z = self.forward(x)
        print('z_mean.shape',z_mean.shape)
        return z_mean[0],z_logvar[0]
    
    def decoder_pass(self,z):
        '''Here we want to look at the latent space of the VRNN, so we get as input only a latent variable of size (1,1,32)'''
        batch_size,_ = z.shape

        h_t = torch.zeros((batch_size, self.rnn_dim)).to(device) # the initial hidden state coudl be different from 0 ... (random ?)

        z_t = z

        # decode
        decoder_input = torch.cat((z_t,h_t),dim=1)
        decoder_output = self.decoder(decoder_input)
        mean_phi_z_t = self.dec_mean(decoder_output) # for now we will not sample from the decoder

        # reconstruction of the observation with convolutional layers
        phi_z_t = mean_phi_z_t.view(-1, self.CNN_channels, 12, 12)
        #print('before deconv',phi_z_t.shape)
        y_t = self.relu(self.dec_conv1(phi_z_t))
        #print('after first deconv',y_t.shape)
        y_t = torch.sigmoid(self.dec_conv2(y_t))
        # print('after second deconv',y_t.shape)

        return y_t