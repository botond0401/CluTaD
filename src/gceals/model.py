import torch.nn as nn
import torch
import torch.nn.functional as F
from .autoencoder import Autoencoder


class GCEALsHead(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.logvar = nn.Parameter(torch.zeros(n_clusters, latent_dim))  # diag cov

    def forward(self, z):
        """
        Computes soft cluster assignments via Mahalanobis distance.
        """
        z_exp = z.unsqueeze(1)  # (B, 1, D)
        mu = self.centroids.unsqueeze(0)  # (1, K, D)
        var = torch.exp(self.logvar).unsqueeze(0)  # (1, K, D)

        dist = ((z_exp - mu) ** 2 / var).sum(dim=2)  # (B, K)
        soft_assignments = F.softmax(-0.5 * dist, dim=1)
        return soft_assignments


class GCEALs(nn.Module):
    def __init__(self, input_dim, latent_dim, n_clusters):
        super().__init__()
        self.ae = Autoencoder(input_dim, latent_dim)
        self.cluster_head = GCEALsHead(latent_dim, n_clusters)

    def forward(self, x):
        z, x_hat = self.ae(x)
        q = self.cluster_head(z)
        return q, x_hat, z

    def pretrain(self, dataloader, optimizer, epochs=50, device='cuda'):
        self.train()
        self.to(device)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            n_samples = 0

            for x_batch in dataloader:
                x_batch = x_batch.to(device)
                _, x_hat = self.ae(x_batch)
                loss = loss_fn(x_hat, x_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x_batch.size(0)
                n_samples += x_batch.size(0)

            avg_loss = total_loss / n_samples
            print(f'[Pretrain] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    def train_gceals(self, dataloader, optimizer, epochs=100, device='cuda', gamma=1.0):
        self.train()
        self.to(device)
        recon_loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            n_samples = 0

            for x_batch in dataloader:
                x_batch = x_batch.to(device)

                q, x_hat, z = self(x_batch)

                # Target distribution p (sharpened version of q)
                weight = q ** 2 / (q.sum(0) + 1e-8)
                p = (weight.T / weight.sum(1)).T

                recon_loss = recon_loss_fn(x_hat, x_batch)
                kl_div = F.kl_div(q.log(), p, reduction='batchmean')

                loss = recon_loss + gamma * kl_div

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x_batch.size(0)
                n_samples += x_batch.size(0)

            avg_loss = total_loss / n_samples
            print(f'[GCEALs] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
