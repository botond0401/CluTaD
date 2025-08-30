import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import torch.nn.utils as nn_utils
import itertools

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

        if self.num_numeric > 0 and pred_num is not None:
            noise_num = noise[:, :self.num_numeric]
            loss_num = ((pred_num - noise_num) ** 2).mean()
        else:
            loss_num = torch.zeros((), device=x.device)

        loss_cat = torch.zeros((), device=x.device)

        has_cats = bool(self.categories) and sum(self.categories) > 0
        if has_cats and pred_cat is not None:
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
        nn_utils.clip_grad_norm_(
            itertools.chain(self.encoder.parameters(), self.denoiser.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        return loss.item(), loss_num.item(), loss_cat.item()


    def pretrain(self, dataloader, optimizer, epochs, batch_size, plot_freq=100):
        """
        Pretrain encoder + denoiser over multiple steps.

        Args:
            dataloader: full dataset tensor (N, D)
            optimizer: optimizer for encoder + denoiser
            epochs: number of pretraining epochs
            batch_size: batch size
            plot_freq: print loss every plot_freq steps
        """
        for epoch in range(epochs):
            total_loss = 0.0
            n_samples = 0
            for (x_batch,) in dataloader:
                if x_batch.ndim == 1:
                    x_batch = x_batch.unsqueeze(0)
                x_batch = x_batch.to(self.device)

                loss, loss_num, loss_cat = self.pretrain_step(x_batch, optimizer)

                total_loss += loss * x_batch.size(0)
                n_samples += x_batch.size(0)

            avg_loss = total_loss / n_samples
            if (epoch+1) % plot_freq == 0:
              print(f'[Pretrain] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')


    def fit_gmm(self, dataloader):
        """
        Fit a Gaussian Mixture Model on the latent space.

        Args:
            x_real: full dataset tensor (N, D)
            n_clusters: number of clusters to fit
        """
        self.encoder.eval()
        latent_z = []
        with torch.no_grad():
            for (x,) in dataloader:
                x = x.to(self.device)
                z_mu, z_sigma2_log = self.encoder(x)
                z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
                latent_z.append(z)
        latent_z = torch.cat(latent_z, 0).detach().cpu().numpy()

        if self.gmm is not None:
            init_weights = self.gmm.weights_
            init_means = self.gmm.means_
            init_precisions = self.gmm.precisions_
            gmm = GaussianMixture(n_components = self.n_clusters,
                                  covariance_type = 'diag',
                                  reg_covar=1e-2,
                                  weights_init = init_weights,
                                  means_init = init_means,
                                  precisions_init = init_precisions)
        else:
          gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag', reg_covar=1e-2)
        gmm.fit(latent_z)
        self.gmm = gmm
        #print(f"✅ GMM fitted with {self.n_clusters} components")


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
        if self.num_numeric > 0 and pred_num is not None:
            noise_num = noise[:, :self.num_numeric]
            rec_loss_num = ((pred_num - noise_num) ** 2).mean()
        else:
            rec_loss_num = torch.zeros((), device=x.device)

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
        pi = torch.tensor(self.gmm.weights_, device=self.device, dtype=torch.float32)
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
        nn_utils.clip_grad_norm_(
            itertools.chain(self.encoder.parameters(), self.denoiser.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        return total_loss.item(), rec_loss.item(), kl_loss.item()



    def train_elbo(self, dataloader, optimizer, batch_size, kl_weight=0.1, plot_freq=100):
        """
        ELBO training loop: combines reconstruction + KL loss.

        Args:
            x_real: dataset (N, D)
            optimizer: optimizer
            batch_size: batch size
            kl_weight: weight on KL
            plot_freq: print every plot_freq steps
        """
        self.encoder.train()
        self.denoiser.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        n_samples = 0

        for (x_batch,) in dataloader:
            if x_batch.ndim == 1:
                x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.to(self.device)
            loss, rec_loss, kl_loss = self.elbo_step(x_batch, optimizer, kl_weight=kl_weight)

            total_loss += loss * x_batch.size(0)
            total_recon_loss += rec_loss * x_batch.size(0)
            total_kl_loss += kl_loss * x_batch.size(0)
            n_samples += x_batch.size(0)

        avg_loss = total_loss / n_samples
        avg_recon_loss = total_recon_loss / n_samples
        avg_kl_loss = total_kl_loss / n_samples
        return avg_loss, avg_recon_loss, avg_kl_loss
