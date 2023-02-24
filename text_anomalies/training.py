import numpy as np
import time
import torch
import fastprogress


def train(dataloader, optimizer, model, loss_fn, device, master_bar):
    """Run one training epoch

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader to use for training
    optimizer : torch.optim.Optimizer
        Optimizer to use for training
    model : torch.nn.Module
        Model to train
    loss_fn : torch.nn.Module
        Loss function to use for training
    device : torch.device
        Device to use for training
    master_bar : fastprogress.master_bar
        Will be iterated over for each epoch to draw batches and display training progress

    Returns
    -------
    epoch_loss : list
        List of loss values for each sample in the training set
    """

    epoch_loss = []
    epoch_similarity = []

    for sentence in fastprogress.progress_bar(dataloader, parent=master_bar):
        optimizer.zero_grad()
        model.train()

        # Move data to device
        sentence = sentence.to(device)

        # Forward pass
        output = model(sentence)

        # Similarity score between the input and the output
        similarity = torch.cosine_similarity(sentence, output, dim=1)
        epoch_similarity.append(similarity.mean().item())

        # Compute loss
        loss = loss_fn(output, sentence)

        # Backward pass
        loss.backward()
        optimizer.step()

        # For plotting the train loss, save it for each sample
        epoch_loss.append(loss.item())

    return np.mean(epoch_loss), np.mean(epoch_similarity)


def validate(dataloader, model, loss_fn, device, master_bar):
    """Compute loss and accuracy on validation set.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader to use for validation
    model : torch.nn.Module
        Model to validate
    loss_fn : torch.nn.Module
        Loss function to use for validation
    device : torch.device
        Device to use for validation
    master_bar : fastprogress.master_bar
        Will be iterated over to draw batches and show validation progress

    Returns
    -------
    epoch_loss : list
        List of loss values for each sample in the validation set
    """

    epoch_loss, epoch_similarity = [], []

    model.eval()
    with torch.no_grad():
        for sentence in fastprogress.progress_bar(dataloader, parent=master_bar):
            # Move data to device
            sentence = sentence.to(device)

            # Forward pass
            output = model(sentence)

            # Similarity score between the input and the output
            similarity = torch.cosine_similarity(sentence, output, dim=1)
            epoch_similarity.append(similarity.mean().item())

            # Compute loss
            loss = loss_fn(output, sentence)

            # For plotting the train loss, save it for each sample
            epoch_loss.append(loss.item())

    return np.mean(epoch_loss), np.mean(epoch_similarity)


def train_model(
    model,
    optimizer,
    loss_function,
    device,
    num_epochs,
    train_dataloader,
    val_dataloader,
    early_stopper=None,
    verbose=False,
    scheduler=None,
):
    """Run model training

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    optimizer : torch.optim.Optimizer
        Optimizer to use for training
    loss_function : torch.nn.Module
        Loss function to use for training
    device : torch.device
        Device to use for training
    num_epochs : int
        Number of epochs to train for
    train_dataloader : torch.utils.data.DataLoader
        Dataloader to use for training
    val_dataloader : torch.utils.data.DataLoader
        Dataloader to use for validation
    early_stopper : EarlyStopping, optional
        Early stopping object to use for training
    verbose : bool, optional
        Whether to print training progress, by default False
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler to use for training, by default None

    Returns
    -------
    train_losses : list
        List of training losses for each epoch
    val_losses : list
        List of validation losses for each epoch
    train_similarities : list
        List of training similarities for each epoch
    val_similarities : list
        List of validation similarities for each epoch
    """

    start_time = time.time()

    master_bar = fastprogress.master_bar(range(num_epochs))
    train_losses, val_losses = [], []
    train_similarities, val_similarities = [], []

    for epoch in master_bar:
        # Train the model
        epoch_train_loss, epoch_train_similarities = train(
            dataloader=train_dataloader,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_function,
            device=device,
            master_bar=master_bar,
        )

        # Validate the model
        epoch_val_loss, epoch_val_similarities = validate(
            dataloader=val_dataloader,
            model=model,
            loss_fn=loss_function,
            device=device,
            master_bar=master_bar,
        )

        # Step the scheduler if given
        if scheduler:
            scheduler.step()

        # Save the losses
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_similarities.append(epoch_train_similarities)
        val_similarities.append(epoch_val_similarities)

        # Print training progress
        if verbose:
            master_bar.write(
                f"Train loss: {np.mean(epoch_train_loss):.4f} | Val loss: {np.mean(epoch_val_loss):.4f}"
            )

        # Early stopping
        if early_stopper:
            early_stopper.update(np.mean(epoch_val_loss), model)
            if early_stopper.early_stop:
                print("Early stopping")
                break

    # Print training time
    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    return train_losses, val_losses, train_similarities, val_similarities
