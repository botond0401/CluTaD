import itertools
from copy import deepcopy
from .training import Trainer, get_diffusion_model
from .denoising_model import MLPDenoiser
from .evaluate import evaluate_mixed_loss


def hyperparameter_search(data_loader, num_classes, num_numerical, device, steps, search_space):
    """
    Performs grid search over hyperparameters for training a diffusion-based denoising model
    on tabular data.

    For each combination of hyperparameters:
    - Initializes an MLP denoiser model
    - Wraps it in a GaussianMultinomialDiffusion object
    - Trains the model using a Trainer class
    - Evaluates it on a validation set using the mixed loss
    - Tracks and saves the best-performing model configuration

    Args:
        data_loader (DataLoader): DataLoader for training data.
        num_classes (List[int]): List of number of classes per categorical feature.
        num_numerical (int): Number of numerical features in the data.
        device (str): Device to use ('cuda' or 'cpu').
        steps (int): Number of training steps per configuration.
        search_space (dict): Dictionary defining the hyperparameter grid, with keys like
                             'd_layers', 'dropout', 'd_t', and 'lr', and list of values for each.

    Returns:
        tuple: A tuple containing:
            - best_config (dict): The hyperparameter setting with the lowest validation loss.
            - best_model_state (dict): The state_dict of the best-performing diffusion model.
            - best_loss (float): The lowest observed validation loss.
    """
    best_loss = float('inf')
    best_loss_random = float('inf')
    best_config = None
    best_model_state = None

    for config in itertools.product(*search_space.values()):
        params = dict(zip(search_space.keys(), config))
        print(f"\n🔍 Trying config: {params}")

        # Create model
        model = MLPDenoiser(
            d_in=num_numerical + sum(num_classes),
            d_layers=params['d_layers'],
            dropout=params['dropout'],
            d_t=params['d_t']
        ).to(device)

        # Wrap with diffusion model
        diffusion = get_diffusion_model(model, num_classes, num_numerical, device)

        # Train
        trainer = Trainer(diffusion, data_loader, device, steps=steps, lr=params['lr'])
        trainer.train()

        # Evaluate
        loss, loss_random = evaluate_mixed_loss(diffusion, data_loader, device)
        print(f"Overall Average Loss: {loss:.4f}, Random Loss: {loss_random:.4f}")

        # Save best
        if loss < best_loss:
            best_loss = loss
            best_loss_random = loss_random
            best_config = params
            best_model_state = deepcopy(diffusion.state_dict())

    return best_config, best_model_state, best_loss, best_loss_random