import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder network containing an enrolled LSTM
    """

    def __init__(self, n_features, hidden_size=256, num_layers=2, **rnn_kwargs):
        super(Encoder, self).__init__()

        # Parameters
        self.n_features = n_features
        self.hidden_size = hidden_size

        # RNN layer
        self.rnn = nn.RNN(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            **rnn_kwargs
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
        _, h_n = self.rnn(x)

        # Return the last hidden state
        # (num_layers, batch, hidden_size)
        return h_n[-1]
