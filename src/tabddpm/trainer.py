import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from .noising_functions import q_sample_num
from .noising_functions import q_sample_cat


class DiffusionTrainer:
    """
    Trainer class for diffusion model on tabular data.
    Handles training loop, loss computation, and plotting.
    """
    def __init__(self, model, optimizer, x_real, num_numeric, categories, T, plot_dir='plots',
                 plot_loss_curve=True, plot_variable_dists=True, plot_freq=100):
        self.model = model
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

    def train(self, steps=500, batch_size=64):
        losses = []

        for step in range(steps):
            idx = torch.randint(0, self.N, (batch_size,))
            x0 = self.x_real[idx]
            t = torch.randint(0, self.T, (batch_size,))

            x0_num = x0[:, :self.num_numeric]
            x0_cat = x0[:, self.num_numeric:]

            x_t_num, noise_num = q_sample_num(x0_num, t, self.T)
            x_t_cat = q_sample_cat(x0_cat, t, self.categories, self.T)

            x_t = torch.cat([x_t_num, x_t_cat], dim=1)
            pred_num, pred_cat = self.model(x_t, t)

            loss_num = ((pred_num - noise_num) ** 2).mean()

            loss_cat = 0.0
            idx_c = 0
            C = len(self.categories)
            for K in self.categories:
                target = x0_cat[:, idx_c:idx_c+K]
                pred_prob = pred_cat[:, idx_c:idx_c+K]
                kl = (target * (torch.log(target + 1e-10) - torch.log(pred_prob + 1e-10))).sum(1).mean()
                loss_cat += kl
                idx_c += K
            loss_cat = loss_cat / C

            loss = loss_num + loss_cat

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.plot_freq == 0:
                print(f"Step {step}: Loss {loss.item():.4f}")
                losses.append(loss.item())

            if self.plot_variable_dists and step % self.plot_freq == 0:
                self.plot_distributions(step)

        if self.plot_loss_curve:
            self.plot_loss(losses, steps)


    def plot_distributions(self, step):
        """Plots distributions of denoised vs original variables."""
        with torch.no_grad():
            t_vis = torch.randint(0, self.T, (self.N,))
            x0_num = self.x_real[:, :self.num_numeric]
            x0_cat = self.x_real[:, self.num_numeric:]

            x_t_num, _ = q_sample_num(x0_num, t_vis, self.T)
            x_t_cat = q_sample_cat(x0_cat, t_vis, self.categories, self.T)

            x_t_full = torch.cat([x_t_num, x_t_cat], dim=1)
            pred_num, pred_cat = self.model(x_t_full, t_vis)

            betas = 0.01 * torch.arange(1, self.T + 1).float() / self.T
            alphas = 1 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)
            alpha_bar_t = alpha_bars[t_vis].unsqueeze(1)
            x0_num_hat = (x_t_num - (1 - alpha_bar_t).sqrt() * pred_num) / alpha_bar_t.sqrt()

            # Numerical plots
            for i in range(self.num_numeric):
                orig_vals = x0_num[:, i].cpu().numpy()
                den_vals = x0_num_hat[:, i].cpu().numpy()
                
                # Compute common bin edges from both datasets combined
                bins = np.histogram_bin_edges(orig_vals, bins=30)

                plt.figure()
                plt.hist(orig_vals, bins=bins, alpha=0.5, label='Original')
                plt.hist(den_vals, bins=bins, alpha=0.5, label='Denoised')
                plt.legend()
                plt.title(f'Step {step} - Numerical var {i}')
                save_dir_num = os.path.join(self.plot_dir, f'num{i}')
                os.makedirs(save_dir_num, exist_ok=True)
                plt.savefig(os.path.join(save_dir_num, f'histogram_step{step}.png'))
                plt.close()

            # Categorical plots
            idx_c = 0
            for j, K in enumerate(self.categories):
                plt.figure()
                orig_cat = torch.argmax(x0_cat[:, idx_c:idx_c+K], dim=1).cpu().numpy()
                pred_cat_argmax = torch.argmax(pred_cat[:, idx_c:idx_c+K], dim=1).cpu().numpy()
                plt.hist(orig_cat, bins=K, alpha=0.5, label='Original')
                plt.hist(pred_cat_argmax, bins=K, alpha=0.5, label='Denoised')
                plt.legend()
                plt.title(f'Step {step} - Categorical var {j}')
                save_dir_cat = os.path.join(self.plot_dir, f'cat{j}')
                os.makedirs(save_dir_cat, exist_ok=True)
                plt.savefig(os.path.join(save_dir_cat, f'histogram_step{step}.png'))
                plt.close()
                idx_c += K


    def plot_loss(self, losses, steps):
        """Plots training loss curve."""
        plt.plot(range(0, steps, 10), losses)
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("Diffusion model training loss")
        plt.savefig(f'{self.plot_dir}/loss_curve.png')
        plt.show()
