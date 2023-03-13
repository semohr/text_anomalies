import torch
import torch.nn as nn
import pytorch_lightning as pl


class Encoder(pl.LightningModule):
    """
    Encoder network containing an enrolled LSTM
    """

    def __init__(self, n_features, hidden_size=256, rnn_num_layers=2, bidirectional=True,):
        super(Encoder, self).__init__()

        # Parameters
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = rnn_num_layers

        # RNN layer
        self.rnn = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=bool(bidirectional),
            dropout=0.25,
        )

    def forward(self, x):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x : torch.Tensor
            The input sequence of shape (batch_size, seq_len, number_of_features).

        Returns
        -------
        h_n : torch.Tensor
            The last hidden state of the LSTM of shape (batch_size, hidden_size).
        """

        # Pass through the LSTM layer
        _ , (h_n,c_n) = self.rnn(x)

        # Unconcat the forward and backward RNN
        #REshape to self.num_layers, 2, batch, self.hidden_size
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
        # Concat the forward and backward RNN
        h_n = torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=1)
        return h_n