import torch
import torch.nn as nn


class Lambda(nn.Module):
    """
    Lambda module converts output of the encoder to latent vector
    """

    def __init__(self, hidden_size, latent_size):
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

        # Layers
        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_size)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_size)

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
