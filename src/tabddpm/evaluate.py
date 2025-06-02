import torch

def evaluate_mixed_loss(dm, data_loader, device):
    dm.eval()
    total_loss_multi = 0.0
    total_loss_gauss = 0.0
    total_loss_multi_random = 0.0
    total_loss_gauss_random = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch, in data_loader:
            x_batch = batch.to(device).float()  # adjust if your loader returns tuple, etc.
            b = x_batch.size(0)
            total_samples += b

            # empty dict if no conditions, else pass your dict
            out_dict = {}

            loss_multi, loss_gauss = dm.mixed_loss(x_batch, out_dict)
            loss_multi_random, loss_gauss_random = dm.mixed_loss(x_batch, out_dict, random=True)

            # loss_multi and loss_gauss are mean per batch, multiply by batch size for sum
            total_loss_multi += loss_multi.item() * b
            total_loss_gauss += loss_gauss.item() * b
            total_loss_multi_random += loss_multi_random.item() * b
            total_loss_gauss_random += loss_gauss_random.item() * b

    return (total_loss_multi + total_loss_gauss) / total_samples, (total_loss_multi_random + total_loss_gauss_random) / total_samples
