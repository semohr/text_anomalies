import torch
import torch.nn as nn
from torch.distributions import Dirichlet
import pytorch_lightning as pl
import torch.nn.functional as F

class Lambda(nn.Module):
    """
    Lambda module converts output of the encoder to latent vector
    """

    def __init__(self, hidden_size, latent_size, bidrectional_rnn=True):
        """
        Parameters
        ----------
        hidden_size : int
            The number of features in the hidden state h of the LSTM.
        latent_size : int
            The size of the latent space.
        """
        super(Lambda, self).__init__()

        # Parameters
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        D = 2 if bidrectional_rnn else 1

        # Layers
        self.hidden_to_mean = nn.Linear(D*self.hidden_size, self.latent_size)
        self.hidden_to_logvar = nn.Linear(D*self.hidden_size, self.latent_size)

        # Initialize the weights
        # p(z) ~ N(0, I)
        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, h_n):
        """
        Get the mean and logvar of the latent space from the cell state of the LSTM.

        Parameters
        ----------
        h_n : torch.Tensor
            The cell state of the LSTM of shape (batch_size, hidden_size).

        Returns
        -------
        z : torch.Tensor
            The latent vector of shape (batch_size, latent_size).
        """

        self.latent_mean = self.hidden_to_mean(h_n)
        self.latent_logvar = self.hidden_to_logvar(h_n)

        # Reparameterization trick
        std = torch.exp(0.5 * self.latent_logvar)
        eps = torch.randn_like(std)
        z = self.latent_mean + eps * std

        return z, self.latent_mean, self.latent_logvar

    def lossfn(self, mu, logvar, reduce=True):
        """
        Compute the KL divergence between the prior and the posterior.

        Returns
        -------
        kl : torch.Tensor
            The KL divergence of shape (1,).
        """
        # KL divergence
        kl = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())

        # Batch mean
        if reduce:
            kl /= mu.size(0)

        return kl



class LambdaDirichlet(pl.LightningModule):
    """
    Lambda module converts output of the encoder to latent vector (dirichlet priors)
    """

    def __init__(self, hidden_size, latent_size, bidrectional_rnn=True, prior_alpha=0.02 ):
        """
        Parameters
        ----------
        hidden_size : int
            The number of features in the hidden state h of the LSTM.
        latent_size : int
            The size of the latent space.
        alpha : float
            The alpha parameter of the dirichlet prior
        """
        super(LambdaDirichlet, self).__init__()

        # Parameters
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.prior_alpha = prior_alpha
        D = 2 if bidrectional_rnn else 1



        # Layers
        self.hidden_to_alpha = nn.Linear(D*self.hidden_size, self.latent_size)

        self.norm = nn.BatchNorm1d(num_features=self.latent_size, eps=0.001, momentum=0.001, affine=True)
        self.norm.weight.data.copy_(torch.ones(self.latent_size))
        self.norm.weight.requires_grad = False


    def forward(self, h_n):
        """
        Get the mean and logvar of the latent space from the cell state of the LSTM.

        Parameters
        ----------
        h_n : torch.Tensor
            The cell state of the LSTM of shape (batch_size, hidden_size).

        Returns
        -------
        z : torch.Tensor
            The latent vector of shape (batch_size, latent_size).
        """
        alpha = self.hidden_to_alpha(h_n)
        alpha = self.norm(alpha)
        alpha = F.softplus(alpha)
        alpha = torch.max(torch.tensor(0.00001, device=self.device), alpha)
        z = rsvi(alpha)
        return z, alpha
    

    def lossfn(self, alpha, reduce=True):
        prior = Dirichlet(torch.ones(alpha.shape, device=self.device) * self.prior_alpha)
        dist = Dirichlet(alpha)
        return torch.distributions.kl.kl_divergence(dist, prior)


def calc_epsilon(p, alpha):
    sqrt_alpha = torch.sqrt(9 * alpha - 3)
    powza = torch.pow(p / (alpha - 1 / 3), 1 / 3)
    return sqrt_alpha * (powza - 1)


def gamma_h_boosted(epsilon, u, alpha):
    """
    Reparameterization for gamma rejection sampler with shape augmentation.
    """
    B = u.shape[0]
    K = alpha.shape[1]
    r = torch.arange(B, device=alpha.device)
    rm = torch.reshape(r, (-1, 1, 1)).float()
    alpha_vec = torch.tile(alpha, (B, 1)).reshape((B, -1, K)) + rm
    u_pow = torch.pow(u, 1. / alpha_vec) + 1e-10
    return torch.prod(u_pow, axis=0) * gamma_h(epsilon, alpha + B)


def gamma_h(eps, alpha):
    b = alpha - 1 / 3
    c = 1 / torch.sqrt(9 * b)
    v = 1 + (eps * c)
    return b * (v ** 3)


def rsvi(alpha):
    B = 10
    gam = torch.distributions.Gamma(alpha + B, 1).sample().to(alpha.device)
    eps = calc_epsilon(gam, alpha + B).detach().to(alpha.device)
    u = torch.rand((B, alpha.shape[0], alpha.shape[1]), device=alpha.device)
    doc_vec = gamma_h_boosted(eps, u, alpha)
    # normalize
    gam = doc_vec
    doc_vec = gam / torch.reshape(torch.sum(gam, dim=1), (-1, 1))
    z = doc_vec.reshape(alpha.shape)
    return z
