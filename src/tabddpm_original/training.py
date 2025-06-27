from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from .gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from .evaluate import evaluate_mixed_loss
from .utils import ohe_to_categories


def get_diffusion_model(model, num_classes, num_numerical, device):
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array(num_classes),
        num_numerical_features=num_numerical,
        denoise_fn=model,
        gaussian_loss_type='mse',
        num_timesteps=10000,
        scheduler='cosine',
        device=device
    )
    diffusion.to(device)
    return diffusion


class Trainer:
    def __init__(self, model, train_loader, device, steps=1000, lr=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.steps = steps
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.ema_model = deepcopy(model._denoise_fn)
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def train(self, log_interval=100):
        step = 0
        losses = []
        overall_losses = []
        reconstruction_dfs = []
        iterator = iter(self.train_loader)

        while step < self.steps:
            try:
                x_batch, = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                x_batch, = next(iterator)

            x_batch = x_batch.to(self.device)

            self.optimizer.zero_grad()
            loss_multi, loss_gauss, _ = self.model.mixed_loss(x_batch, {})
            loss = loss_multi + loss_gauss
            loss.backward()
            self.optimizer.step()

            overall_loss, _, loss_info_list = evaluate_mixed_loss(self.model, self.train_loader, self.device)

            if step % 100 == 0:
                print(f"[{step}/{self.steps}] Loss: {loss.item():.4f}")

            losses.append(loss.item())
            overall_losses.append(overall_loss)

            if step % log_interval == 0:

                with torch.no_grad():
                    num_num = self.model.num_numerical_features
                    num_cat = len(self.model.num_classes)

                    all_x_num_recon = []
                    all_x_cat_recon = []

                    for loss_info in loss_info_list:
                        model_out = loss_info['model_out']
                        t = loss_info['t']
                        x_num_t = loss_info['x_num_t']
                        log_x_cat_t = loss_info['log_x_cat_t']

                        model_out_num = model_out[:, :num_num]
                        model_out_cat = model_out[:, num_num:]

                        x_num_recon = self.model._predict_xstart_from_eps(x_num_t, t, model_out_num)
                        log_x_cat_recon = self.model.p_sample(model_out_cat, log_x_cat_t, t, out_dict={})
                        x_cat_recon = ohe_to_categories(torch.exp(log_x_cat_recon), self.model.num_classes)

                        all_x_num_recon.append(x_num_recon.cpu())
                        all_x_cat_recon.append(x_cat_recon.cpu())

                    x_num_all = torch.cat(all_x_num_recon, dim=0)
                    x_cat_all = torch.cat(all_x_cat_recon, dim=0)

                    df = pd.DataFrame()
                    for i in range(num_num):
                        df[f'num_{i}_recon'] = x_num_all[:, i].numpy()
                    for i in range(num_cat):
                        df[f'cat_{i}_recon'] = x_cat_all[:, i].numpy()
                    df['step'] = step

                    reconstruction_dfs.append(df)

            step += 1

        return losses, overall_losses, reconstruction_dfs
