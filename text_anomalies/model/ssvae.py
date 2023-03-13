import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import TypedDict

from .encoder import Encoder
from .decoder import Decoder
from .lambd import Lambda, LambdaDirichlet


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
        rnn_num_layers: int = 2,
        learning_rate: float = 1e-4,
        nu1=0.1,
        nu2=20,
    ) -> None:
        """
        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary. I.e. the number of existing unique tokens.
        label_size : int
            The size of the y vector. I.e. the number of existing unique classes.
        embedding_size : int
            The size of the embedding. Default: 300.
        hidden_dim : int
            The number of features in the hidden state h of the LSTM. Default: 256.
        latent_dim : int
            The dimension of the latent space. Default: 16.
        rnn_num_layers : int
            The number of layers in the Rnn. Default: 2.
        learning_rate : float
            The learning rate. Default: 1e-4.
        nu1 : float
            The first hyperparameter of the loss function.
        nu2 : float
            The second hyperparameter of the loss function.
        """
        super(SSVAE, self).__init__()

        # Parameters
        self.vocab_size = vocab_size
        self.label_size = label_size

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.rnn_num_layers = rnn_num_layers
        self.learning_rate = learning_rate
        self.nu1 = nu1
        self.nu2 = nu2

        # Layers i.e. subnetworks
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_size,
        )

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
            n_features=int(self.embedding_size * 0.5),
            hidden_size=self.hidden_size,
            rnn_num_layers=self.rnn_num_layers,
            bidirectional=True,
        )

        self.decoder = Decoder(
            latent_size=self.latent_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size,
            rnn_num_layers=self.rnn_num_layers,
            bidirectional=True,
        )

        self.y_predict = nn.Sequential(
            nn.Linear(self.latent_size,  self.label_size),
            nn.ReLU(),
            nn.Linear(self.label_size, self.label_size),
            nn.ReLU(),
        )

        # Hidden to latent space
        self.lambd = LambdaDirichlet(
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            bidrectional_rnn=True,
        )


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

        # Highway network
        x_embed = self.input_highway(x_embed)
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
        z, alpha = self.lambd(  # (batch_size, latent_size)
            h_n=h_n,
        )

        y_hat = self.y_predict(z)

        # Decoder
        x_hat = self.decoder(
            z=z,
            seq_len=seq_len,
        )
        x_hat = x_hat.log_softmax(dim=2)

        return x_hat, y_hat, alpha

    def lossfn(self, x_hat, x_true, y_hat, y_true, alpha):
        """
        Loss function for the model.

        Parameters
        ----------
        x : torch.Tensor
            The input sequence of shape (batch_size, seq_len).
        y : torch.Tensor
            The input sequence of shape (batch_size, num_aux_features).
        """

        # Category Loss
        catloss = nn.functional.cross_entropy(
            input=y_hat, target=y_true
        )

        # Reconstruction Loss
        x_hat = x_hat.permute(0, 2, 1)  # (batch_size, vocab_size, seq_len)
        reconloss = nn.functional.nll_loss(
            input=x_hat, target=x_true
        )  # (batch_size, seq_len)

        kldloss = self.lambd.lossfn(alpha, reduce=False)

        # ´loss=recons_loss+0.1*cat_loss+epoch*0.001*kl_loss

        return reconloss, catloss, kldloss

    """
    ----------------------------------------------------------------
    Training
    ----------------------------------------------------------------
    """

    def _common_step(self, batch, batch_idx):
        # Get input, target and labels
        x = batch["x"]
        x_true = batch["x_true"]
        y_true = batch["y_true"]
        y_true = y_true.type(torch.LongTensor)
        y_true = y_true.to(self.device)

        # Forward pass
        x_hat, y_hat, alpha = self.forward(x)

        # Loss
        reconloss, catloss, kldloss = self.lossfn(
            x_hat, x_true, y_hat, y_true, alpha
        )
        loss = reconloss + self.nu1 * catloss + (self.nu2 * kldloss).mean()

        # Batch mean
        loss = loss.mean()

        acc_seq = self.accuracy_sequence(x_hat, x_true)
        acc_labels = self.accuracy_label(y_hat, y_true)

        return loss, (acc_labels, acc_seq)

    def training_step(self, batch, batch_idx):
        loss, (acc_labels, acc_seq) = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, (acc_labels, acc_seq) = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return {
            "val_loss": loss,
            "acc_labels": acc_labels,
            "acc_seq": acc_seq,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"].mean() for x in outputs]).mean()
        avg_acc_labels = torch.stack([x["acc_labels"].mean() for x in outputs]).mean()
        avg_acc_seq = torch.stack([x["acc_seq"].mean() for x in outputs]).mean()
        self.log("val_acc_labels", avg_acc_labels, on_epoch=True, prog_bar=True)
        self.log("val_acc_seq", avg_acc_seq, on_epoch=True, prog_bar=True)
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)

    def accuracy_label(self, y_hat, y):
        """
        Measures the distance between the true labels and the predicted labels.
        """
        y_hat = torch.argmax(y_hat, dim=1)
        accuracy = torch.sum(y_hat == y, dim=0) / y.size(0)
        return accuracy

    def accuracy_sequence(self, x_hat, x):
        """
        Compute the accuracy for sequences we ignore all padding.
        """
        x_hat = torch.argmax(x_hat, dim=2)
        accuracy = torch.sum(x_hat == x, dim=1) / torch.sum(x != 0, dim=1)
        return accuracy

    def configure_optimizers(self):
        """
        Configure the optimizers.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            The optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
