import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


class ClusterDDPM:
    """
    ClusterDDPM model: wraps encoder, denoiser, diffusion schedules, and GMM.
    Supports pretraining, ELBO training, GMM fitting, and sampling.
    """
    def __init__(self, encoder, denoiser, T, num_numeric, categories, n_clusters, device):
        self.encoder = encoder
        self.denoiser = denoiser
        self.T = T
        self.num_numeric = num_numeric
        self.categories = categories
        self.n_clusters = n_clusters
        self.device = device

        # Diffusion schedules
        betas = 0.01 * torch.arange(1, T + 1).float() / T
        alphas = 1 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0).to(device)
        self.sqrtab = self.alpha_bars.sqrt()
        self.sqrtmab = (1 - self.alpha_bars).sqrt()

        self.gmm = None  # Will hold fitted GMM

    def pretrain_step(self, x, optimizer):
        """
        One pretraining step: predict noise from x_t + z + t
        """
        B = x.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=self.device) - 1
        noise = torch.randn_like(x)

        x_t = self.sqrtab[t].unsqueeze(1) * x + self.sqrtmab[t].unsqueeze(1) * noise

        mu, logvar = self.encoder(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()

        t_norm = t.float() / self.T
        pred_num, pred_cat = self.denoiser(x_t, z, t_norm)

        noise_num = noise[:, :self.num_numeric]
        loss_num = ((pred_num - noise_num) ** 2).mean()

        loss_cat = 0.0
        x0_cat = x[:, self.num_numeric:]
        idx_c = 0
        for K in self.categories:
            target = x0_cat[:, idx_c:idx_c+K]
            pred_prob = pred_cat[:, idx_c:idx_c+K]
            kl = (target * (torch.log(target + 1e-10) - torch.log(pred_prob + 1e-10))).sum(1).mean()
            loss_cat += kl
            idx_c += K
        if len(self.categories) > 0:
            loss_cat /= len(self.categories)

        loss = loss_num + loss_cat

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), loss_num.item(), loss_cat.item()


    def pretrain(self, x_real, optimizer, steps=500, batch_size=64, plot_freq=100):
        """
        Pretrain encoder + denoiser over multiple steps.

        Args:
            x_real: full dataset tensor (N, D)
            optimizer: optimizer for encoder + denoiser
            steps: number of training steps
            batch_size: batch size
            plot_freq: print loss every plot_freq steps
        """
        N = x_real.shape[0]
        losses = []

        for step in range(steps):
            idx = torch.randint(0, N, (batch_size,))
            x_batch = x_real[idx]

            loss, loss_num, loss_cat = self.pretrain_step(x_batch, optimizer)

            if step % plot_freq == 0:
                print(f"Step {step}: Total {loss:.4f} | Num {loss_num:.4f} | Cat {loss_cat:.4f}")
                losses.append(loss)

        return losses
    
        
    def fit_gmm(self, x_real):
        """
        Fit a Gaussian Mixture Model on the latent space.

        Args:
            x_real: full dataset tensor (N, D)
            n_clusters: number of clusters to fit
        """
        self.encoder.eval()
        latent_list = []

        with torch.no_grad():
            N = x_real.shape[0]
            for i in range(0, N):
                mu, logvar = self.encoder(x_real)
                z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                latent_list.append(z.cpu())

        latent_z = torch.cat(latent_list, dim=0).numpy()

        # Fit GMM
        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag')
        gmm.fit(latent_z)
        self.gmm = gmm
        print(f"✅ GMM fitted with {self.n_clusters} components")


    def elbo_step(self, x, optimizer, kl_weight=0.1):
        """
        One ELBO training step for ClusterDDPM.
        
        Args:
            x: input batch (B, D)
            optimizer: optimizer
            kl_weight: weight for the KL terms
        
        Returns:
            total_loss, rec_loss, kl_loss
        """
        B = x.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=self.device) - 1
        noise = torch.randn_like(x)

        # Diffusion forward process
        x_t = self.sqrtab[t].unsqueeze(1) * x + self.sqrtmab[t].unsqueeze(1) * noise

        # Encoder
        mu_phi, logvar_phi = self.encoder(x)
        sigma2_phi = torch.exp(logvar_phi)
        z = mu_phi + torch.randn_like(mu_phi) * (0.5 * logvar_phi).exp()

        # Denoising
        t_norm = t.float() / self.T
        pred_num, pred_cat = self.denoiser(x_t, z, t_norm)

        # Reconstruction loss
        noise_num = noise[:, :self.num_numeric]
        rec_loss_num = ((pred_num - noise_num) ** 2).mean()

        rec_loss_cat = 0.0
        x0_cat = x[:, self.num_numeric:]
        idx_c = 0
        for K in self.categories:
            target = x0_cat[:, idx_c:idx_c + K]
            pred_prob = pred_cat[:, idx_c:idx_c + K]
            kl = (target * (torch.log(target + 1e-10) - torch.log(pred_prob + 1e-10))).sum(1).mean()
            rec_loss_cat += kl
            idx_c += K
        if len(self.categories) > 0:
            rec_loss_cat /= len(self.categories)

        rec_loss = rec_loss_num + rec_loss_cat

        # GMM prior
        pi = torch.from_numpy(self.gmm.weights_).to(x).float()
        c_mu = torch.from_numpy(self.gmm.means_).to(x).float()
        c_var = torch.from_numpy(self.gmm.covariances_).to(x).float()
        c_logvar = torch.log(c_var)

        # p(c|z) = soft responsibilities
        det = 1e-10
        log_prob_c = []
        for c in range(self.gmm.n_components):
            logp = -0.5 * (
                torch.sum(np.log(2 * np.pi) + c_logvar[c] +
                        (z - c_mu[c]) ** 2 / c_var[c], dim=1)
            )
            log_prob_c.append((torch.log(pi[c] + det) + logp).unsqueeze(1))
        log_prob_c = torch.cat(log_prob_c, dim=1)
        logsumexp = torch.logsumexp(log_prob_c, dim=1)
        w_c = torch.exp(log_prob_c - logsumexp.unsqueeze(1))  # (B, K)

        # Cat KL: -λ sum_c w_c log (pi / w_c)
        cat_kl = - (w_c * (torch.log(pi + det) - torch.log(w_c + det))).sum(1).mean()

        # Gaussian KL
        gauss_kl = 0.5 * (
            w_c[:, :, None].squeeze(2) * (
                c_logvar[None, :, :] +
                sigma2_phi[:, None, :] / c_var[None, :, :] +
                (mu_phi[:, None, :] - c_mu[None, :, :]) ** 2 / c_var[None, :, :]
            ).sum(2)
        ).sum(1).mean()

        # Variational entropy term
        entropy = -0.5 * (1 + logvar_phi).sum(1).mean()

        kl_loss = cat_kl + gauss_kl + entropy

        # Total loss
        total_loss = rec_loss + kl_weight * kl_loss

        # Backward + update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss.item(), rec_loss.item(), kl_loss.item()

    

    def train_elbo(self, x_real, optimizer, steps=1000, batch_size=64, kl_weight=0.1, plot_freq=100):
        """
        ELBO training loop: combines reconstruction + KL loss.

        Args:
            x_real: dataset (N, D)
            optimizer: optimizer
            steps: number of steps
            batch_size: batch size
            kl_weight: weight on KL
            plot_freq: print every plot_freq steps
        """
        N = x_real.shape[0]
        losses = []

        for step in range(steps):
            idx = torch.randint(0, N, (batch_size,))
            x_batch = x_real[idx]

            total_loss, rec_loss, kl_loss = self.elbo_step(x_batch, optimizer, kl_weight=kl_weight)

            if step % plot_freq == 0:
                print(f"Step {step}: Total {total_loss:.4f} | Rec {rec_loss:.4f} | KL {kl_loss:.4f}")
                losses.append(total_loss)

        return losses


