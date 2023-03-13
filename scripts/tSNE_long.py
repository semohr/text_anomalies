import torch
import os
import numpy as np
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE

import text_anomalies as ta
from tqdm.auto import tqdm
from multiprocessing import Pool

# Load data
data = ta.DOECDataModule(data_dir="../data/doec/")
data.prepare_data()
data.setup()
dataset = data.dataset
dataloader = DataLoader(
    dataset=dataset,
    batch_size=64,
    num_workers=os.cpu_count(),
    shuffle=False,
    collate_fn=dataset.collate_fn,
)

# Load model
model = ta.model.SSVAE.load_from_checkpoint(
    "../data/doec/models/ssvae_4.ckpt",
    vocab_size=30_000,
    label_size=data.num_classes,
    latent_size=65,
    hidden_size=512,
    embedding_size=300,
    rnn_num_layers=5,
)
model.eval()
model.to("cuda:3")

# Get embeddings
embeddings = []
for batch in tqdm(dataloader):
    x = batch["x"].to("cuda:3")

    # Forward pass and save latent space
    x_hat, y_hat, alpha = model(x)

    # Save embeddings
    embeddings.append(alpha.detach().cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)


# Perform t-SNE with different perplexities
def pTSNE(embeddings, perplexity):
    tsne = TSNE(
        n_components=2,
        verbose=1,
        n_iter=5000,
        perplexity=perplexity,
        random_state=42,
        early_exaggeration=18,
    )
    tsne_results = tsne.fit_transform(embeddings)

    # Save t-SNE results
    np.save(f"../data/doec/tsne_results_{perplexity}.npy", tsne_results)


with Pool(processes=8) as pool:

    # Perform t-SNE
    pool.starmap(
        pTSNE,
        [
            (embeddings, 310),
            (embeddings, 350),
            (embeddings, 400),
            (embeddings, 450),
            (embeddings, 500),
        ],
    )
