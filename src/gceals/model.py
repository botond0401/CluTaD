import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


class GCEALsHead(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

        # Centroids μ_j (to be initialized with k-means, Algorithm 1 step 6)
        self.centroids = nn.Parameter(torch.zeros(n_clusters, latent_dim))

        # Diagonal covariance Σ_j = diag(exp(logvar))
        self.logvar = nn.Parameter(torch.zeros(n_clusters, latent_dim))

        # MLP Head for auxiliary distribution Q (Eq. 10)
        self.mlp = nn.Linear(latent_dim, n_clusters)

    def forward(self, z):
        """
        Returns:
          P = cluster distribution from Gaussian/Mahalanobis (Eq. 6–9)
          Q = cluster distribution from MLP head (Eq. 10)
        """
        if z.dim() == 1:  # single sample (D,)
            z = z.unsqueeze(0)  # → (1, D)

        B, D = z.shape

        # --- Gaussian assignment (P) ---
        # Expand z to (B, K, D)
        z_exp = z.unsqueeze(1).expand(B, self.n_clusters, D)  # (B, K, D)
        mu = self.centroids.unsqueeze(0).expand(B, self.n_clusters, D)  # (B, K, D)
        var = torch.exp(self.logvar).unsqueeze(0).expand(B, self.n_clusters, D)  # (B, K, D)

        dist = torch.sqrt(((z_exp - mu) ** 2 / var).sum(dim=2))  # (B, K)
        P = F.softmax(-dist, dim=1)  # soft assignments across clusters

        # --- MLP assignment (Q) ---
        logits = self.mlp(z)  # (B, K)
        Q = F.softmax(logits, dim=1)

        return P, Q


class GCEALs(nn.Module):
    def __init__(self, input_dim, latent_dim, n_clusters):
        super().__init__()
        self.ae = Autoencoder(input_dim, latent_dim)
        self.cluster_head = GCEALsHead(latent_dim, n_clusters)
        self.n_clusters = n_clusters
        self.pi = None  # cluster priors (Eq. 8)

    def forward(self, x):
        z, x_hat = self.ae(x)
        P, Q = self.cluster_head(z)
        return P, Q, x_hat, z

    def pretrain(self, dataloader, optimizer, epochs=50, device='cuda', patience=10):
        """
        Pretrain autoencoder only (Eq. 11).
        """
        self.train()
        self.to(device)
        loss_fn = nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            total_loss, n_samples = 0.0, 0

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

            # --- Early stopping check ---
            if avg_loss < best_loss - 1e-4:  # small tolerance
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'[Pretrain] Early stopping at epoch {epoch+1}, best loss={best_loss:.4f}')
                break

    def init_centroids(self, dataloader, device='cuda'):
        """
        Initialize centroids μ_j using k-means on latent space (Alg. 1, step 6).
        """
        self.eval()
        all_z = []
        with torch.no_grad():
            for x_batch in dataloader:
                x_batch = x_batch.to(device)
                z, _ = self.ae(x_batch)
                all_z.append(z.cpu())
        all_z = torch.cat(all_z, dim=0).numpy()

        # Run k-means
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        kmeans.fit(all_z)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)

        # Initialize cluster centroids μ_j
        self.cluster_head.centroids.data.copy_(centroids)

        # Initialize priors ω_j equally (Eq. 8)
        self.pi = torch.ones(self.n_clusters, device=device) / self.n_clusters
        print("[Init] Centroids initialized with k-means")

    def train_gceals(self, dataloader, optimizer, epochs=100, device='cuda', gamma=1.0):
        """
        Joint training of AE + clustering (Algorithm 1).
        Loss = L_rec + γ * KL(P || Q)   (Eq. 13)
        """
        self.train()
        self.to(device)
        recon_loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss, n_samples = 0.0, 0

            for x_batch in dataloader:
                x_batch = x_batch.to(device)

                # Forward pass
                P, Q, x_hat, z = self(x_batch)  # P=Gaussian, Q=MLP

                # ====== Loss ======
                recon_loss = recon_loss_fn(x_hat, x_batch)
                kl_div = F.kl_div(Q.log(), P, reduction='batchmean')  # KL(P || Q)
                loss = recon_loss + gamma * kl_div

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x_batch.size(0)
                n_samples += x_batch.size(0)

            # ====== Update priors ω_j (Eq. 8) ======
            with torch.no_grad():
                all_P = []
                for x_batch in dataloader:
                    x_batch = x_batch.to(device)
                    P, _, _, _ = self(x_batch)
                    all_P.append(P)
                all_P = torch.stack(all_P, dim=0)
                self.pi = all_P.mean(dim=0)

            avg_loss = total_loss / n_samples
            print(f'[GCEALs] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

            # ====== Early stopping check (Algorithm 1) ======
            if torch.any(self.pi <= 1.0 / (2 * self.n_clusters)):
                print(f"[Early Stopping] Cluster prior too small: {self.pi}")
                break
