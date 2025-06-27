import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularEncoder(nn.Module):
    """
    Encoder for tabular data.
    Maps input features to latent mean and log-variance for Gaussian latent space.
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
            
        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        h = self.feature_extractor(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
