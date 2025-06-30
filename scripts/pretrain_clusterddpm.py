import torch
import torch.optim as optim
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clusterddpm.encoder import TabularEncoder
from src.clusterddpm.denoiser import Denoiser
from src.clusterddpm.pretrainer import ClusterDDPMTrainer

# Config
DATA_PATH = 'data/preprocessed/1480/data_processed.csv'
PLOTS_PATH = 'plots/clusterddpm/1480'
CHECKPOINT_PATH = 'models/1480/clusterddpm/pretrain_checkpoint.pth'

num_numeric = 9
categories = [2]
T = 200
steps = 1000
batch_size = 32
dim_hidden = 64
latent_dim = 10
lr = 1e-3

# Load data
df = pd.read_csv(DATA_PATH)
x_real = torch.tensor(df.values, dtype=torch.float32)
N, D = x_real.shape

# Models
encoder = TabularEncoder(
    input_dim=D,
    hidden_dims=[64, 64],
    latent_dim=latent_dim
)

denoiser = Denoiser(
    dim_in=D,
    latent_dim=latent_dim,
    dim_hidden=dim_hidden,
    num_numeric=num_numeric,
    categories=categories
)

# Optimizer
optimizer = optim.Adam(
    list(encoder.parameters()) + list(denoiser.parameters()),
    lr=lr
)

# Trainer
trainer = ClusterDDPMTrainer(
    encoder=encoder,
    denoiser=denoiser,
    optimizer=optimizer,
    x_real=x_real,
    num_numeric=num_numeric,
    categories=categories,
    T=T,
    plot_dir=PLOTS_PATH,
    plot_loss_curve=True,
    plot_variable_dists=True,
    plot_freq=100
)

# Run pretraining
trainer.train(steps=steps, batch_size=batch_size)

# Save checkpoint
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
torch.save({
    'encoder': encoder.state_dict(),
    'denoiser': denoiser.state_dict(),
    'optimizer': optimizer.state_dict(),
    'T': T,
    'num_numeric': num_numeric,
    'categories': categories
}, CHECKPOINT_PATH)
print(f"✅ Pretraining checkpoint saved at {CHECKPOINT_PATH}")
