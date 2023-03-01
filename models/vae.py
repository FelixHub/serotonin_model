import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        
        self.fc_mu = nn.Linear(2*2*256, latent_dim)
        self.fc_logsigma = nn.Linear(2*2*256, latent_dim)


    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_dim, 256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction

class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_dim,beta = 1):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_dim)
        self.decoder = Decoder(img_channels, latent_dim)
        self.beta = beta

    def forward(self, x): # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
        

    def loss_function(self, recon_x, x, mu, logsigma):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
        return BCE + self.beta*KLD, KLD
    
    def encoder_pass(self,x):
        return self.encoder(x)
    
    def decoder_pass(self,z):
        return self.decoder(z)