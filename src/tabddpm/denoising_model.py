from torch import nn
from src.tabddpm.modules import timestep_embedding, MLP


class MLPDenoiser(nn.Module):
    """
    MLP-based denoising network for diffusion models on tabular data.

    This model predicts the noise added to the original data during the diffusion process.
    It uses a projection layer to inject timestep information into the input, followed by
    an MLP to regress the noise.

    Args:
        d_in (int): Input feature dimension.
        d_layers (list[int]): Hidden layer dimensions for the MLP.
        dropout (float): Dropout rate applied in the MLP.
        d_t (int): Dimensionality for the timestep embedding and projection space.
    """
    def __init__(self, d_in, d_layers, dropout, d_t):
        super().__init__()
        self.dim_t = d_t

        # MLP that outputs the predicted noise (same shape as input)
        self.mlp = MLP.make_baseline(d_t, d_layers, dropout, d_in)

        # Linear projection from input feature space to embedding space
        self.proj = nn.Linear(d_in, d_t)

        # Timestep embedding network (maps sinusoidal embedding to same dimension)
        self.time_embed = nn.Sequential(
            nn.Linear(d_t, d_t),
            nn.SiLU(),
            nn.Linear(d_t, d_t)
        )

    def forward(self, x, timesteps):
        """
        Forward pass of the MLPDenoiser.

        Args:
            x (Tensor): Noised input data of shape (batch_size, d_in).
            timesteps (Tensor): Timesteps tensor of shape (batch_size,) representing the diffusion step.

        Returns:
            Tensor: Predicted noise of shape (batch_size, d_in), matching the input shape.
        """
        # Embed the diffusion timestep and transform via a small MLP
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))

        # Inject timestep embedding into the input feature space
        x = self.proj(x) + emb

        # Predict the added noise using the MLP
        return self.mlp(x)
