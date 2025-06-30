import os
import torch
import matplotlib.pyplot as plt
import numpy as np


class ClusterDDPMTrainer:
    """
    Trainer class for ClusterDDPM pretraining on tabular data.
    Trains encoder + denoiser to predict noise from noised inputs.
    """
    def __init__(self, encoder, denoiser, optimizer, x_real, num_numeric, categories, T,
                 plot_dir='plots', plot_loss_curve=True, plot_variable_dists=True, plot_freq=100):
        self.encoder = encoder
        self.denoiser = denoiser
        self.optimizer = optimizer
        self.x_real = x_real
        self.num_numeric = num_numeric
        self.categories = categories
        self.T = T
        self.N, self.D = x_real.shape
        self.plot_dir = plot_dir
        self.plot_loss_curve = plot_loss_curve
        self.plot_variable_dists = plot_variable_dists
        self.plot_freq = plot_freq

        os.makedirs(plot_dir, exist_ok=True)

        # Precompute schedules
        betas = 0.01 * torch.arange(1, T + 1).float() / T
        alphas = 1 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0).to(x_real.device)
        self.sqrtab = self.alpha_bars.sqrt()
        self.sqrtmab = (1 - self.alpha_bars).sqrt()

    def train(self, steps=500, batch_size=64):
        losses = []

        for step in range(steps):
            idx = torch.randint(0, self.N, (batch_size,))
            x0 = self.x_real[idx]

            t = torch.randint(0, self.T, (batch_size,), device=x0.device)
            noise = torch.randn_like(x0)

            x_t = self.sqrtab[t].unsqueeze(1) * x0 + self.sqrtmab[t].unsqueeze(1) * noise

            mu, logvar = self.encoder(x0)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()

            t_norm = t.float() / self.T
            pred_num, pred_cat = self.denoiser(x_t, z, t_norm)

            # Numerical loss
            noise_num = noise[:, :self.num_numeric]
            loss_num = ((pred_num - noise_num) ** 2).mean()

            # Categorical loss
            loss_cat = 0.0
            x0_cat = x0[:, self.num_numeric:]
            idx_c = 0
            C = len(self.categories)
            for K in self.categories:
                target = x0_cat[:, idx_c:idx_c+K]
                pred_prob = pred_cat[:, idx_c:idx_c+K]
                kl = (target * (torch.log(target + 1e-10) - torch.log(pred_prob + 1e-10))).sum(1).mean()
                loss_cat += kl
                idx_c += K
            if C > 0:
                loss_cat = loss_cat / C

            loss = loss_num + loss_cat

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.plot_freq == 0:
                print(f"Step {step}: Loss {loss.item():.4f}")
                losses.append(loss.item())
                if self.plot_variable_dists:
                    self.plot_distributions(step)

        if self.plot_loss_curve:
            self.plot_loss(losses)

    def plot_distributions(self, step):
        with torch.no_grad():
            t_vis = torch.randint(0, self.T, (self.N,), device=self.x_real.device)
            noise = torch.randn_like(self.x_real)
            x_t = self.sqrtab[t_vis].unsqueeze(1) * self.x_real + self.sqrtmab[t_vis].unsqueeze(1) * noise

            mu, logvar = self.encoder(self.x_real)
            z = mu  # mean for visualization

            t_norm = t_vis.float() / self.T
            pred_num, pred_cat = self.denoiser(x_t, z, t_norm)

            # Reconstruct numerical x0
            alpha_bar_t = self.alpha_bars[t_vis].unsqueeze(1)
            x0_num = self.x_real[:, :self.num_numeric]
            x_t_num = x_t[:, :self.num_numeric]
            x0_num_hat = (x_t_num - (1 - alpha_bar_t).sqrt() * pred_num) / alpha_bar_t.sqrt()

            # Plot numerical
            for i in range(self.num_numeric):
                orig_vals = x0_num[:, i].cpu().numpy()
                den_vals = x0_num_hat[:, i].cpu().numpy()
                bins = np.histogram_bin_edges(orig_vals, bins=30)

                plt.figure()
                plt.hist(orig_vals, bins=bins, alpha=0.5, label='Original')
                plt.hist(den_vals, bins=bins, alpha=0.5, label='Denoised')
                plt.legend()
                plt.title(f'Step {step} - Numerical var {i}')
                save_dir = os.path.join(self.plot_dir, f'num{i}')
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'histogram_step{step}.png'))
                plt.close()

            # Plot categorical
            x0_cat = self.x_real[:, self.num_numeric:]
            idx_c = 0
            for j, K in enumerate(self.categories):
                orig_cat = torch.argmax(x0_cat[:, idx_c:idx_c+K], dim=1).cpu().numpy()
                pred_cat_argmax = torch.argmax(pred_cat[:, idx_c:idx_c+K], dim=1).cpu().numpy()

                plt.figure()
                plt.hist(orig_cat, bins=K, alpha=0.5, label='Original')
                plt.hist(pred_cat_argmax, bins=K, alpha=0.5, label='Denoised')
                plt.legend()
                plt.title(f'Step {step} - Categorical var {j}')
                save_dir = os.path.join(self.plot_dir, f'cat{j}')
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'histogram_step{step}.png'))
                plt.close()
                idx_c += K

    def plot_loss(self, losses):
        plt.plot(np.arange(0, len(losses)) * self.plot_freq, losses)
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("ClusterDDPM pretraining loss")
        plt.savefig(os.path.join(self.plot_dir, "pretrain_loss_curve.png"))
        plt.show()
