import torch
import torch.nn as nn
import torch.nn.functional as F


class Denoiser(nn.Module):
    """
    Simple denoiser model for tabular diffusion.
    Applies a linear -> ReLU -> linear architecture.
    Outputs numerical predictions and categorical probabilities (via softmax).
    """
    def __init__(self, dim_in, latent_dim, dim_hidden, num_numeric, categories):
        super().__init__()
        self.num_numeric = num_numeric
        self.categories = categories
        self.net = nn.Sequential(
            nn.Linear(dim_in + latent_dim + 1, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_in)
        )


    def forward(self, x, z, t):
        """
        Forward pass of the denoiser.

        Args:
            x: Tensor, shape (batch_size, dim_in)
            t: Tensor, shape (batch_size,)

        Returns:
            out_num: numerical denoised output
            out_cat: categorical probabilities after softmax
        """
        t = t.unsqueeze(1).float()
        xzt = torch.cat([x, z, t], dim=1)
        out = self.net(xzt)

        out_num = out[:, :self.num_numeric]
        out_cat_raw = out[:, self.num_numeric:]

        out_cat = []
        idx = 0
        for K in self.categories:
            logits = out_cat_raw[:, idx:idx+K]
            probs = F.softmax(logits, dim=1)
            out_cat.append(probs)
            idx += K

        out_cat = torch.cat(out_cat, dim=1) if out_cat else None
        return out_num, out_cat
