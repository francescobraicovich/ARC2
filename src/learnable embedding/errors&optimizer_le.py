import torch
import torch.nn.functional as F
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

loss_env = F.max_overlap_loss(noisy_next_state, predicted_next_state)
loss_act = F.cross_entropy(reconstructed_action_logits, action_idx)  # replaced F.action_loss with cross_entropy

total_loss = loss_env + loss_act



