import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import TypedDict

from .encoder import Encoder
from .decoder import Decoder
from .lambd import Lambda


class SSVAEConfig(TypedDict):
    """
    Configuration of the self supervised variational autoencoder
    """

    embedding_size: int
    hidden_size: int
    latent_size: int


class SSVAE(pl.LightningModule):
    """
    Self supervised variational autoencoder
    """

    def __init__(
        self,
        vocab_size: int,
        label_size: int,
        embedding_size: int = 300,
        hidden_size: int = 256,
        latent_size: int = 16,
    ) -> None:
        """
        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary. I.e. the number of existing unique tokens.
        label_size : int
            The size of the y vector. I.e. the number of existing unique labels.
        embedding_size : int
            The size of the embedding. Default: 300.
        hidden_dim : int
            The number of features in the hidden state h of the LSTM. Default: 256.
        latent_dim : int
            The dimension of the latent space. Default: 16.
        lstm_num_layers : int
            The number of layers in the LSTM. Default: 2.
        """
        super(SSVAE, self).__init__()

        # Parameters
        self.vocab_size = vocab_size
        self.label_size = label_size

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Layers i.e. subnetworks
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_size,
        )
        self.encoder = Encoder(
            n_features=self.embedding_size,
            hidden_size=self.hidden_size,
        )
        self.pre_decoder = nn.Sequential(
            nn.Linear(self.latent_size + self.label_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.latent_size),
        )
        self.decoder = Decoder(
            latent_size=self.latent_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size,
            device=self.device,
        )

        self.y_predict = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.label_size),
        )

        # Hidden to latent space
        self.lambd = Lambda(
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
        )

        # Initialize the weights
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input sequence of shape (batch_size, seq_len).
        y : torch.Tensor
            The input sequence of shape (batch_size, num_aux_features).

        Returns
        -------
        recons : torch.Tensor
            The reconstructed sequence of shape (batch_size, seq_len).
        zmu : torch.Tensor
            The mean of the latent space of shape (batch_size, latent_size).
        zlogvar : torch.Tensor
            The logvar of the latent space of shape (batch_size, latent_size).
        ysoft : torch.Tensor
            The soft label of shape (batch_size, num_aux_features).
        """

        # Embed the input
        x_embed = self.embedding(x)
        seq_len = x_embed.size(1)

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

        y_hat = self.y_predict(h_n)

        # Predecoder i.e. combining z and y
        z = self.pre_decoder(torch.cat((z, y_hat), dim=1))

        # Decoder
        x_hat = self.decoder(
            z=z,
            seq_len=seq_len,
        )
        x_hat = x_hat.log_softmax(dim=2)

        return x_hat, y_hat, z_mu, z_logvar

    def loss(self, x, y):
        """
        Loss function for the model.

        Parameters
        ----------
        x : torch.Tensor
            The input sequence of shape (batch_size, seq_len).
        y : torch.Tensor
            The input sequence of shape (batch_size, num_aux_features).
        """

        # Encode the input
        x_hat, y_hat, z_mu, z_logvar = self.forward(x)

        # Category Loss
        y_hat = y_hat.log_softmax(dim=1)
        catloss = nn.functional.cross_entropy(input=y_hat, target=y)

        # Reconstruction Loss
        x_hat = x_hat.permute(0, 2, 1)  # (batch_size, vocab_size, seq_len)
        reconloss = nn.functional.nll_loss(
            input=x_hat, target=x
        )  # (batch_size, seq_len)

        kldloss = torch.mean(
            -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0
        )

        # ´loss=recons_loss+0.1*cat_loss+epoch*0.001*kl_loss

        return reconloss, catloss, kldloss

    """
    ----------------------------------------------------------------
    Training
    ----------------------------------------------------------------
    """

    def _common_step(self, batch, batch_idx):
        # Get input, target and labels
        x, x_true, y_true = batch

        # Forward pass
        x_hat, y_hat, z_mu, z_logvar = self.forward(x)

        # Loss
        reconloss, catloss, kldloss = self.loss(
            x_hat, x_true, y_hat, y_true, z_mu, z_logvar
        )

        loss = reconloss + 0.1 * catloss + self.current_epoch * 0.001 * kldloss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizers.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            The optimizer.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=10, verbose=True, min_lr=1e-6
        )
        return optimizer
