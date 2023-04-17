import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def gmm_loss(batch,mus,sigmas,logpi):

    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)

    return - torch.mean(log_prob)


class MDRNN(nn.Module):

    def __init__(self, latent_dim, action_dim, hidden_dim, gaussians_nb,memory ='rnn'):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gaussians_nb = gaussians_nb

        self.memory = memory

        self.gmm_linear = nn.Linear(hidden_dim, (2 * latent_dim + 1) * gaussians_nb ) # +2 for reward and termination but we don't care here

        if memory == 'rnn':
            self.rnn = nn.RNN(latent_dim + action_dim, hidden_dim)
        else : 
            self.rnn = nn.LSTM(latent_dim + action_dim, hidden_dim)

    def forward(self,action,latent,hidden):

        in_al = torch.cat([action,latent],dim=-1)

        if self.memory == 'rnn':
            next_hidden = self.rnn(in_al.unsqueeze(0), hidden.unsqueeze(0))
            out_rnn = next_hidden[0].squeeze(0)
        else :
            out = self.rnn(in_al.unsqueeze(0), hidden)
            out_rnn = out[0].squeeze(0)
            next_hidden = out[1]


        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians_nb * self.latent_dim

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians_nb, self.latent_dim)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.gaussians_nb, self.latent_dim)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians_nb]
        pi = pi.view(-1, self.gaussians_nb)
        logpi = F.log_softmax(pi, dim=-1)

        return mus, sigmas, logpi, next_hidden

    def loss_function(self,mus, sigmas, logpi,latent_next_obs):
            
        loss = gmm_loss(latent_next_obs,mus,sigmas,logpi)

        return loss