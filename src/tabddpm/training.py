from copy import deepcopy
import numpy as np
import torch.optim as optim
from .gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from .evaluate import evaluate_mixed_loss


def get_diffusion_model(model, num_classes, num_numerical, device):
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array(num_classes),
        num_numerical_features=num_numerical,
        denoise_fn=model,
        gaussian_loss_type='mse',
        num_timesteps=1000,
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

    def train(self):
        step = 0
        losses = []
        overall_losses = []
        iterator = iter(self.train_loader)

        while step < self.steps:
            try:
                x_batch, = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                x_batch, = next(iterator)

            x_batch = x_batch.to(self.device)

            self.optimizer.zero_grad()
            loss_multi, loss_gauss = self.model.mixed_loss(x_batch, {})
            loss = loss_multi + loss_gauss
            loss.backward()
            self.optimizer.step()

            overall_loss, _ = evaluate_mixed_loss(self.model, self.train_loader, self.device)

            if step % 100 == 0:
                print(f"[{step}/{self.steps}] Loss: {loss.item():.4f}")
            losses.append(loss.item())
            overall_losses.append(overall_loss)

            step += 1
        return losses, overall_losses
