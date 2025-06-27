import torch


def q_sample_num(x0_num, t, T):
    """
    Forward diffusion for numerical features (Gaussian).

    Returns noised sample and the true noise used.
    """
    noise = torch.randn_like(x0_num)
    betas = 0.01 * torch.arange(1, T + 1).float() / T
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)
    alpha_bar_t = alpha_bars[t].unsqueeze(1)
    x_t_num = alpha_bar_t.sqrt() * x0_num + (1 - alpha_bar_t).sqrt() * noise
    return x_t_num, noise


def q_sample_cat(x0_cat, t, categories, T):
    """
    Forward diffusion for categorical features (multinomial).

    Returns noised soft one-hot vectors.
    """
    beta = 0.01 * t.float() / T
    beta = beta.unsqueeze(1)
    x_t_cat = []
    idx = 0
    for K in categories:
        x0_slice = x0_cat[:, idx:idx+K]
        noisy = (1 - beta) * x0_slice + beta / K
        x_t_cat.append(noisy)
        idx += K
    return torch.cat(x_t_cat, dim=1)