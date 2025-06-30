import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clusterddpm.encoder import TabularEncoder
from src.clusterddpm.denoiser import Denoiser
from src.clusterddpm.model import ClusterDDPM
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

# Config
DATA_PATH = 'data/preprocessed/1480/data_processed.csv'
PLOTS_PATH = 'plots/clusterddpm/1480'
CHECKPOINT_PATH_PRE = 'models/1480/clusterddpm/pretrain_checkpoint.pth'
CHECKPOINT_PATH_FINAL = 'models/1480/clusterddpm/final_checkpoint.pth'

num_numeric = 9
categories = [2]
T = 200
pretrain_steps = 1000
em_epochs = 10
m_steps = 100
batch_size = 32
dim_hidden = 64
latent_dim = 10
lr = 1e-3
n_clusters = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(DATA_PATH)
x_real = torch.tensor(df.values, dtype=torch.float32).to(device)
N, D = x_real.shape

# Models
encoder = TabularEncoder(
    input_dim=D,
    hidden_dims=[64, 64],
    latent_dim=latent_dim
).to(device)

denoiser = Denoiser(
    dim_in=D,
    latent_dim=latent_dim,
    dim_hidden=dim_hidden,
    num_numeric=num_numeric,
    categories=categories
).to(device)

# ClusterDDPM wrapper
model = ClusterDDPM(
    encoder=encoder,
    denoiser=denoiser,
    T=T,
    num_numeric=num_numeric,
    categories=categories,
    n_clusters=2,
    device=device
)

# Optimizer
optimizer = optim.Adam(
    list(encoder.parameters()) + list(denoiser.parameters()),
    lr=lr
)

# 🔹 Pretraining
print("🔹 Starting pretraining...")
model.pretrain(x_real, optimizer, steps=pretrain_steps, batch_size=batch_size, plot_freq=100)

# Save pretraining checkpoint
os.makedirs(os.path.dirname(CHECKPOINT_PATH_PRE), exist_ok=True)
torch.save({
    'encoder': encoder.state_dict(),
    'denoiser': denoiser.state_dict(),
    'optimizer': optimizer.state_dict(),
    'T': T,
    'num_numeric': num_numeric,
    'categories': categories
}, CHECKPOINT_PATH_PRE)
print(f"✅ Pretraining checkpoint saved at {CHECKPOINT_PATH_PRE}")

# 🔹 Fit initial GMM (E-step 0)
print("🔹 Fitting initial GMM...")
model.fit_gmm(x_real)

# 🔹 EM training loop
print("🔹 Starting EM training...")
for epoch in range(em_epochs):
    print(f"🌟 EM epoch {epoch+1}/{em_epochs}")
    
    # M-step
    model.train_elbo(x_real, optimizer, steps=m_steps, batch_size=batch_size, kl_weight=0.1, plot_freq=50)
    
    # E-step
    model.fit_gmm(x_real)

# Save final model
os.makedirs(os.path.dirname(CHECKPOINT_PATH_FINAL), exist_ok=True)
torch.save({
    'encoder': encoder.state_dict(),
    'denoiser': denoiser.state_dict(),
    'optimizer': optimizer.state_dict(),
    'gmm': model.gmm
}, CHECKPOINT_PATH_FINAL)
print(f"✅ Final checkpoint saved at {CHECKPOINT_PATH_FINAL}")




# Encode all data and compute GMM assignments
with torch.no_grad():
    mu, logvar = model.encoder(x_real)
    z_np = mu.cpu().numpy()
    y_pred = model.gmm.predict(z_np)

# Load ground truth
y_true = pd.read_csv(DATA_PATH.replace("data_processed.csv", "clusters.csv")).values.flatten()

# Define cluster alignment function
def cluster_accuracy(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = dict(zip(col_ind, row_ind))
    y_aligned = np.array([mapping[label] for label in y_pred])
    acc = accuracy_score(y_true, y_aligned)
    return acc, y_aligned

# Compute metrics
accuracy, y_aligned = cluster_accuracy(y_true, y_pred)
ari = adjusted_rand_score(y_true, y_pred)

# Print results
print(f"✅ Final clustering performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ARI: {ari:.4f}")
