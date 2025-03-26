import torch
import torch.nn.functional as F
from torch.optim import AdamW
from enviroment import maximum_overlap_regions
# Loss for the environment model (Equation (5)):
# Here, we use MSELoss as an example for continuous next-state prediction.

# Loss for the action reconstruction (Equation (6)):
# We use cross-entropy loss to compare the logits with the true action.

def max_overlap_loss(original_grid, reconstructed_grid):
    # Calculate the maximum overlap score between the original and reconstructed grids.
    _, _, overlap_score = maximum_overlap_regions(original_grid, reconstructed_grid)
    # Our loss should increase as overlap decreases so we define the loss as 1 - overlap_score.
    return 1.0 - overlap_score

loss_env = F.max_overlap_loss(actual_next_state, predicted_next_state)

loss_act = F.cross_entropy(logits, action_idx)  # replaced F.action_loss with cross_entropy

total_loss = loss_env + loss_act

def get_grad_norm(parameters):
    """
    Compute the global 2-norm of gradients for a list of parameters.
    Useful for logging/troubleshooting exploding gradients.
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def update_with_adamw(model, optimizer, loss, max_grad_norm=5.0):
    """
    Update model parameters using AdamW optimizer with gradient clipping.
    
    Args:
        model: The neural network model to update
        optimizer: AdamW optimizer instance
        loss: The loss to backpropagate
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        grad_norm: The gradient norm before clipping
    """
    # Zero gradients
    optimizer.zero_grad()
    
    # Backpropagate loss
    loss.backward()
    
    # Calculate gradient norm for logging
    grad_norm = get_grad_norm(model.parameters())
    
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    # Update parameters
    optimizer.step()
    
    return grad_norm

# Example usage:
# Initialize model and optimizer
# model = YourModel()
# optimizer = AdamW(model.parameters(), 
#                   lr=1e-4, 
#                   weight_decay=1e-2,  # AdamW's weight decay parameter
#                   betas=(0.9, 0.999))
# MAX_GRAD_NORM = 5.0

# In training loop:
# loss = calculate_loss(...)
# grad_norm = update_with_adamw(model, optimizer, loss, MAX_GRAD_NORM)
# log_metrics(loss=loss.item(), grad_norm=grad_norm)
