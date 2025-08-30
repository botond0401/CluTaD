import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import adjusted_rand_score
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gceals.model import GCEALs
from src.utils import cluster_accuracy


dataset_index = '23'

# Config
DATA_PATH = f'data/preprocessed/{dataset_index}/data_processed.csv'
LABEL_PATH = f'data/preprocessed/{dataset_index}/clusters.csv'
METADATA_PATH = f'data/preprocessed/{dataset_index}/metadata.json'


with open(METADATA_PATH, 'r') as f:
  metadata = json.load(f)
num_numeric = metadata['num_numerical_features']
categories = metadata['num_classes_per_cat']
n_clusters = metadata['num_clusters']

pretrain_epochs = 100
train_epochs = 50
batch_size = 256
lr = 1e-3
gamma=0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
x_real = pd.read_csv(DATA_PATH).values.astype(np.float32)
y_true = pd.read_csv(LABEL_PATH).values.flatten()
if y_true.dtype.kind in {'U', 'S', 'O'}:
    unique_labels, y_true = np.unique(np.asarray(y_true).astype(str), return_inverse=True)
x_real = torch.tensor(x_real, dtype=torch.float32).to(device)
dataset = TensorDataset(x_real)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
N, D = x_real.shape



BEST_PATH = f"best_checkpoint_{dataset_index}.pth"
best_acc = -1.0
best_meta = None
for latent_dim in [5, 10, 15, 20]:
  print(f"latent_dim: {latent_dim}")
  # Models
  model = GCEALs(input_dim=D, latent_dim=latent_dim, n_clusters=n_clusters).to(device)


  # Optimizer
  optimizer = optim.Adam(model.parameters(), lr=lr)

  # Pretrain AE
  print("🔹 Pretraining autoencoder...")
  model.pretrain(x_real, optimizer, epochs=pretrain_epochs, device=device)

  # Full training
  print("🔹 Training GCEALs model...")
  model.train_gceals(x_real, optimizer, epochs=train_epochs, device=device, gamma=gamma)

  # Predict cluster assignments
  model.eval()
  with torch.no_grad():
      z, _ = model.ae(x_real)
      _, q = model.cluster_head(z)
      y_pred = q.argmax(dim=1).cpu().numpy()

  # Alignment
  accuracy, y_aligned = cluster_accuracy(y_true, y_pred)
  ari = adjusted_rand_score(y_true, y_pred)

  print(f"✅ Final clustering performance:")
  print(f"Accuracy: {accuracy:.4f}")
  print(f"ARI: {ari:.4f}")


  # Example: after you finish one training run
  results = {
      "accuracy": float(accuracy),      # your computed accuracy
      "ari": float(ari),                # your computed ARI
      "latent_dim": int(latent_dim),    # latent dimension
      "dataset_index": str(dataset_index)      # e.g., "mnist", "cifar10", etc.
  }

  # Path to results file
  results_file = Path("gceals_results.json")

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
          "latent_dim": int(latent_dim),
          "dataset_index": str(dataset_index),
          "accuracy": float(accuracy),
          "ari": float(ari)
      }

      torch.save({
          "model_state": model.state_dict(),                # AE + cluster head
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "config": {
                "input_dim": D,
                "latent_dim": latent_dim,
                "n_clusters": n_clusters,
                "lr": lr,
                "batch_size": batch_size,
                "pretrain_epochs": pretrain_epochs,
                "gceals_epochs": train_epochs,
                "gamma": gamma,
                "architecture": {
                    "encoder_layers": [D, 500, 500, 2000, latent_dim],
                    "decoder_layers": [latent_dim, 2000, 500, 500, D],
                    },
                },
            "metrics": {
                "accuracy": float(accuracy),
                "ari": float(ari)
                }
            }, BEST_PATH)
    
      print(f"💾 New best model saved to {BEST_PATH} (acc={accuracy:.4f}, z={latent_dim}, K={n_clusters})")
    # ----------------------------------------------------------

# --- after both loops finish ---
print("🏁 Tuning finished.")
print(f"Best acc: {best_acc:.4f}")
if best_meta is not None:
    print(f"Best config: latent_dim={best_meta['latent_dim']}")
