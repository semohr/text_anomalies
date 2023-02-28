import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .lambd import Lambda


class VAE(torch.nn.Module):
    """
    Variational Autoencoder (VAE) with LSTM encoder and decoder.
    """

    def __init__(
        self,
        vocab_size,
        embedding_size=300,
        hidden_size=256,
        latent_size=16,
        device="cpu",
    ):
        """
        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary. I.e. the number of existing unique tokens.
        embedding_size : int
            The size of the embedding. Default: 300.
        hidden_dim : int
            The number of features in the hidden state h of the LSTM. Default: 256.
        latent_dim : int
            The dimension of the latent space. Default: 16.
        lstm_num_layers : int
            The number of layers in the LSTM. Default: 2.
        """
        super(VAE, self).__init__()

        # Parameters
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.input_highway = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, int(self.embedding_size * 0.8)),
            nn.ReLU(),
            nn.Linear(int(self.embedding_size * 0.8), int(self.embedding_size * 0.6)),
            nn.ReLU(),
            nn.Linear(int(self.embedding_size * 0.6), int(self.embedding_size * 0.5)),
            nn.ReLU(),
        )

        self.encoder = Encoder(
            n_features=int(self.embedding_size * 0.5), hidden_size=self.hidden_size
        )

        self.lambd = Lambda(hidden_size=self.hidden_size, latent_size=self.latent_size)
        self.decoder = Decoder(
            latent_size=self.latent_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size,
            device=device,
        )

        # Initialize the weights
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.input_highway[0].weight)
        torch.nn.init.xavier_uniform_(self.input_highway[2].weight)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, seq_len). Padding is expected.

        Returns
        -------
        x_hat : torch.Tensor
            The output tensor of shape (batch_size, seq_len, vocab_size).

        """

        #  Embed the input
        x_embed = self.embedding(x)  # (batch_size, seq_len, embedding_size)
        seq_len = x_embed.size(1)

        # Highway
        x_embed = self.input_highway(x_embed)  # (batch_size, seq_len, embedding_size)

        # Pack the padded sequence
        packed_x_embed = torch.nn.utils.rnn.pack_padded_sequence(
            input=x_embed,
            lengths=torch.sum(x != 0, dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Encoder
        h_n = self.encoder(  # (batch_size, hidden_size)
            x=packed_x_embed,
        )

        # Lambda i.e. Reparameterization for the latent space
        z, z_mu, z_logvar = self.lambd(  # (batch_size, latent_size)
            h_n=h_n,
        )

        # Decoder
        x_hat = self.decoder(
            z=z,
            seq_len=seq_len,
        )

        x_hat = x_hat.log_softmax(dim=2)

        return x_hat, (z, z_mu, z_logvar)

    def loss_function(self, x_hat, x, z_mu, z_logvar, coef=1.0):
        """
        Reconstruction + KL divergence losses summed over all elements and batch

        Parameters
        ----------
        x_hat : torch.Tensor
            The output tensor of shape (batch_size, seq_len, vocab_size). Containing the
            probabilities of each token in the vocabulary.
        x : torch.Tensor
            The input tensor of shape (batch_size, seq_len). Containing the tokenized
            sentences.
        z_mu : torch.Tensor
            The mean of the latent space of shape (batch_size, latent_size).
        z_logvar : torch.Tensor
            The log variance of the latent space of shape (batch_size, latent_size).

        Returns
        -------
        loss : torch.Tensor
            The loss of the model.

        """
        # Reconstruction Loss
        x_hat = x_hat.permute(0, 2, 1)  # (batch_size, vocab_size, seq_len)
        BCE = nn.functional.nll_loss(input=x_hat, target=x)  # (batch_size, seq_len)

        KLD = torch.mean(
            -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0
        )

        return BCE + coef * KLD, coef * KLD, BCE
