import torch
import torch.optim as optim
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tabddpm.diffusion_model import Denoiser
from src.tabddpm.trainer import DiffusionTrainer


DATA_PATH = 'data/preprocessed/1480/data_processed.csv'
PLOTS_PATH = 'plots/tabddpm/1480'
num_numeric = 9
categories = [2]
T = 200
steps = 1000
batch_size = 32
dim_hidden = 64
lr = 1e-3

df = pd.read_csv(DATA_PATH)
x_real = torch.tensor(df.values, dtype=torch.float32)
N, D = x_real.shape

model = Denoiser(D, dim_hidden, num_numeric, categories)
optimizer = optim.Adam(model.parameters(), lr=lr)

trainer = DiffusionTrainer(
    model=model,
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

trainer.train(steps=steps, batch_size=batch_size)
