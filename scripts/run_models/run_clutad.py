import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.clutad.encoder import TabularEncoder
from src.clutad.denoiser import Denoiser
from src.clutad.model import CluTaD
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader, TensorDataset
from src.utils import cluster_accuracy


dataset_index = '40975'


# Config
DATA_PATH = f'data/preprocessed/{dataset_index}/data_processed.csv'
LABEL_PATH = f'data/preprocessed/{dataset_index}/clusters.csv'
METADATA_PATH = f'data/preprocessed/{dataset_index}/metadata.json'


with open(METADATA_PATH, 'r') as f:
  metadata = json.load(f)
num_numeric = metadata['num_numerical_features']
categories = metadata['num_classes_per_cat']
n_clusters = metadata['num_clusters']

T = 100 # this is a question !!!

pretrain_steps = 1000 # same as in example
em_epochs = 1000 # same as in example (it is 1000 altogether)
batch_size = 256 # same as in example
hidden_dims=[500, 500, 2000] # same as in example
kl_weight=0.1 # same as in example
lr = 1e-3 # same as in example



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(DATA_PATH)
x_real = torch.tensor(df.values, dtype=torch.float32).to(device)
dataset = TensorDataset(x_real)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Load ground truth
y_true = pd.read_csv(LABEL_PATH).values.flatten()
if y_true.dtype.kind in {'U', 'S', 'O'}:
    unique_labels, y_true = np.unique(np.asarray(y_true).astype(str), return_inverse=True)
N, D = x_real.shape

BEST_PATH = f"best_checkpoint_{dataset_index}.pth"
best_acc = -1.0
best_meta = None
for dim_hidden in [500, 1000]:
  for latent_dim in [5, 10, 15, 20]:
    print(f"dim_hidden {dim_hidden}, latent_dim: {latent_dim}")

    # Models
    encoder = TabularEncoder(
        input_dim=D,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim
    ).to(device)

    denoiser = Denoiser(
        dim_in=D,
        latent_dim=latent_dim,
        dim_hidden=dim_hidden,
        num_numeric=num_numeric,
        categories=categories
    ).to(device)

    # CluTaD wrapper
    model = CluTaD(
        encoder=encoder,
        denoiser=denoiser,
        T=T,
        num_numeric=num_numeric,
        categories=categories,
        n_clusters=n_clusters,
        device=device
    )

    # Optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(denoiser.parameters()) +
        list(model.mlp.parameters()),
        lr=lr, weight_decay=1e-4
    )

    # 🔹 Pretraining
    print("🔹 Starting pretraining...")
    model.pretrain(dataloader, optimizer, epochs=pretrain_steps, batch_size=batch_size, plot_freq=100)

    # 🔹 Fit initial GMM (E-step 0)
    print("🔹 Fitting initial GMM...")
    model.fit_gmm(dataloader)

    # Optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(denoiser.parameters()) +
        list(model.mlp.parameters()),
        lr=lr, weight_decay=1e-4
    )

    # 🔹 EM training loop
    print("🔹 Starting EM training...")
    for epoch in range(em_epochs):
        avg_loss, avg_recon_loss, avg_cluster_loss, stop = model.train_elbo(
            dataloader, optimizer, batch_size=batch_size, kl_weight=kl_weight, plot_freq=50
        )
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{em_epochs}, "
                  f"Loss: {avg_loss:.4f}, Recon-Loss: {avg_recon_loss:.4f}, "
                  f"Cluster-Loss: {avg_cluster_loss:.4f}")

        #if stop:
            #print(f"⏹️ Stopping early at epoch {epoch+1}")
            #break

        if epoch % 10 == 0:
          # E-step
            model.fit_gmm(dataloader)


    # Encode all data and compute GMM assignments
    with torch.no_grad():
        mu, logvar = model.encoder(x_real)
        z_np = mu.cpu().numpy()
        y_pred = model.gmm.predict(z_np)

    # Compute metrics
    accuracy, y_aligned = cluster_accuracy(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    # Print results
    print(f"✅ Final clustering performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ARI: {ari:.4f}")


    # Example: after you finish one training run
    results = {
        "accuracy": float(accuracy),      # your computed accuracy
        "ari": float(ari),                # your computed ARI
        "T": int(T),                      # diffusion timesteps
        "dim_hidden": int(dim_hidden),    # hidden dimension
        "latent_dim": int(latent_dim),    # latent dimension
        "dataset_index": str(dataset_index)      # e.g., "mnist", "cifar10", etc.
    }

    # Path to results file
    results_file = Path("clutad_results.json")

    # If file exists, load and append; else create new
    if results_file.exists():
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(results)

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved results to {results_file}\n")

    # --- NEW: update 'best' and save checkpoint if improved ---
    if accuracy > best_acc:
        best_acc = float(accuracy)
        best_meta = {
            "T": int(T),
            "latent_dim": int(latent_dim),
            "dim_hidden": int(dim_hidden),
            "dataset_index": str(dataset_index),
            "accuracy": float(accuracy),
            "ari": float(ari)
        }

        # save immediately so you don't lose it if the script stops
        torch.save({
            "encoder": encoder.state_dict(),
            "denoiser": denoiser.state_dict(),
            "optimizer": optimizer.state_dict(),  # optional but handy
            "gmm": model.gmm,                     # sklearn object; torch.save pickles it
            "config": {
                "T": T,
                "num_numeric": num_numeric,
                "categories": categories,
                "n_clusters": n_clusters,
                "hidden_dims": hidden_dims,
                "dim_hidden": dim_hidden,
                "latent_dim": latent_dim,
                "kl_weight": kl_weight,
                "lr": lr,
                "batch_size": batch_size,
                "em_epochs": em_epochs,
                "pretrain_steps": pretrain_steps,
            },
            "metrics": {
                "accuracy": float(accuracy),
                "ari": float(ari)
            }
        }, BEST_PATH)

        print(f"💾 New best model saved to {BEST_PATH} (acc={accuracy:.4f}, T={T}, z={latent_dim})")
    # ----------------------------------------------------------

# --- after both loops finish ---
print("🏁 Tuning finished.")
print(f"Best acc: {best_acc:.4f}")
if best_meta is not None:
    print(f"Best config: dim_hidden={best_meta['dim_hidden']}, latent_dim={best_meta['latent_dim']}")
