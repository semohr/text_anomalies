import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_size, embedding_size, hidden_size, vocab_size, device):
        """
        Parameters"""
        super(Decoder, self).__init__()

        # Parameters:
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device
        self.num_layers = 2

        # Layers
        self.latent_to_hidden = nn.Linear(
            self.latent_size, self.hidden_size * self.num_layers
        )
        self.rnn = nn.RNN(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            nonlinearity="tanh",
        )
        self.output = nn.Linear(self.hidden_size, self.vocab_size)

        # Initialize the weights
        torch.nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        torch.nn.init.xavier_uniform_(self.output.weight)

    def forward(self, z, seq_len):
        """
        Parameters
        ----------
        z : torch.Tensor
            The latent vector of shape (batch_size, latent_dim).
        seq_len : torch.Tensor
            The sequence length of shape (batch_size,).

        Returns
        -------
        x_hat : torch.Tensor

        """
        hidden = self.latent_to_hidden(z)  # (batch_size, hidden_dim * num_layers)
        hidden = hidden.view(
            self.num_layers, -1, self.hidden_size
        )  # (num_layers, batch_size, hidden_dim)
        input = torch.zeros(
            (z.shape[0], seq_len, self.embedding_size), device=self.device
        )  # (batch_size, seq_len, vocab_size )
        output = self.rnn(input, hidden)

        x_hat = self.output(output[0])

        return x_hat
