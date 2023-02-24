import numpy as np
import torch


class EarlyStopper:
    """Early stops the training if validation accuracy does not increase after a
    given patience.
    """

    def __init__(self, verbose=False, path="checkpoint.pt", patience=4):
        """Initialization.

        Parameters
        ----------
        verbose : bool, optional
            Print additional information. Defaults to False.
        path: str, optional
            Path where checkpoints should be saved.
            Defaults to 'checkpoint.pt'.
        patience : int, optional
                Number of epochs to wait for increasing
                accuracy. If accyracy does not increase, stop training early.
                Defaults to 4.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.__early_stop = False
        self.val_acc_max = -np.Inf
        self.path = path
        self.num_epochs = 0

    @property
    def early_stop(self):
        """True if early stopping criterion is reached.

        Returns
        -------
        stop: bool
            True if early stopping criterion is reached.
        """
        return self.__early_stop

    @property
    def best_acc(self):
        """Best accuracy achieved so far.

        Returns
        -------
        best_acc: float
            Best accuracy achieved so far.
        """
        return self.val_acc_max

    @property
    def best_epoch(self):
        """Epoch number of best accuracy.

        Returns
        -------
        best_epoch: int
            Epoch number of best accuracy.
        """
        return self._best_epoch

    def update(self, val_acc, model):
        """Call after one epoch of model training to update early stopper object.

        Parameters
        ----------
        val_acc : float
            Accuracy on validation set.
        model : nn.Module
            torch model that is trained.
        """

        # Check if accuracy increased
        self.num_epochs = self.num_epochs + 1  # Epoch number of best acc
        if val_acc > self.val_acc_max:
            self.save_checkpoint(model, val_acc)
            self.val_acc_max = val_acc
            self.counter = 0
            self._best_epoch = self.num_epochs
        else:
            self.counter = self.counter + 1

        # If counter reached patience stop!
        if self.counter == self.patience:
            self.load_checkpoint(model)
            self.__early_stop = True

    def save_checkpoint(self, model, val_acc):
        """Save model checkpoint.

        Parameters
        ----------
        model : nn.Module
             Model of which parameters should be saved.
        """
        if self.verbose:
            print(
                f"Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)

    def load_checkpoint(self, model):
        """Load model from checkpoint.

        Parameters
        ----------
        model : nn.Module
            Model that should be reset to parameters loaded from checkpoint.

        Returns
        -------
            model : nn.Module
                Model with parameters from checkpoint
        """
        if self.verbose:
            print(
                f"Loading model from last checkpoint with validation accuracy {self.val_acc_max:.6f}"
            )

        model.load_state_dict(torch.load(self.path))
        return model
