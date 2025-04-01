import os
import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from utils.util import set_device
from world_model.memory_data_load import WorldModelDataset

from dsl.utilities.plot import save_grid, to_numpy

# Determine the device: CUDA -> MPS -> CPU
DEVICE = set_device('world_model/train_world_model.py')

def loss_fn(state_logits, shape_logits, state, shape):
    """
    Custom loss function for the world model.
    
    Args:
        state_logits (torch.Tensor): Predicted state logits (shape: [batch, seq_len, num_classes]).
        shape_logits (torch.Tensor): Predicted shape logits (shape: [batch, seq_len, num_classes]).
        state (torch.Tensor): Actual state (shape: [batch, seq_len]).
        shape (torch.Tensor): Actual shape (shape: [batch, seq_len]).

    Returns:
        torch.Tensor: Computed loss value.
    """
    # Resize the logits to match the shape expected by nn.CrossEntropyLoss.
    # Expected input shape: [batch, num_classes, seq_len]
    state_logits = state_logits.permute(0, 2, 1)
    shape_logits = shape_logits.permute(0, 2, 1)
    
    # Standard cross entropy losses for state and shape predictions.
    criterion = nn.CrossEntropyLoss()
    state_loss = criterion(state_logits, state)
    shape_loss = criterion(shape_logits, shape)
    
    # Create a mask that is 1 for elements where state is not 0, 0 otherwise.
    mask = (state != 0).float()

    # Compute additional cross entropy loss only for cells where state != 0.
    # We use reduction="none" to get elementwise loss values.
    criterion_none = nn.CrossEntropyLoss(reduction="none")
    # The output from criterion_none has shape [batch, seq_len].
    masked_loss = criterion_none(state_logits, state)
    # Apply the mask
    masked_loss = masked_loss * mask
    # Average over non-zero positions. Avoid division by zero.
    if mask.sum() > 0:
        non_padded_loss = masked_loss.sum() / mask.sum()
    else:
        non_padded_loss = torch.tensor(0.0, device=state.device)
    
    # Combine the losses
    total_loss = state_loss + shape_loss + non_padded_loss
    return total_loss, state_loss, shape_loss, non_padded_loss

def world_model_train(
    state_encoder,
    action_embedder,
    transition_model,
    world_model_args,
    save_model_dir,
    logger=None,
    save_per_epochs=1,
    eval_interval=1,
    plot_interval=1
):
    """
    Train the world model using state encoding, action embedding, and a transition model.
    """
    epochs = world_model_args.get('epochs', 10)
    lr = world_model_args.get('lr', 1e-3)
    batch_size = world_model_args.get('batch_size', 32)
    max_iter = world_model_args.get('max_iter', None)
    # Early stopping parameters
    patience = world_model_args.get('early_stopping_patience', 3)
    best_eval_non_padded_loss = float('inf')
    patience_counter = 0

    # Create optimizer for all network parameters
    optimizer = torch.optim.Adam(
        list(state_encoder.parameters()) +
        list(action_embedder.parameters()) +
        list(transition_model.parameters()),
        lr=lr
    )

    # Loss function for shape and state prediction
    criterion = nn.CrossEntropyLoss()

    # Load the dataset and create train/test splits
    dataset = WorldModelDataset()
    train_set, test_set = dataset.train_test_split(test_ratio=0.2)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    action_embedding_num_params = action_embedder.num_parameters
    state_encoder_num_params = state_encoder.num_parameters
    transition_model_num_params = transition_model.num_parameters

    print('-' * 50)
    logger.info('Starting World Model Training')
    logger.info(f"Action Embedding Parameters: {action_embedding_num_params}")
    logger.info(f"State Encoder Parameters: {state_encoder_num_params}")
    logger.info(f"Transition Model Parameters: {transition_model_num_params}")

    global_step = 0
    for epoch in range(epochs):
        state_encoder.train()
        action_embedder.train()
        transition_model.train()
        
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_losses = []
        shape_losses = []
        state_losses = []
        non_padded_losses = []
        for batch in progress_bar:
            # Transfer data to the device
            current_state = batch['current_state'].to(DEVICE)
            current_shape = batch['current_shape'].to(DEVICE)
            next_current_state = batch['target_state'].to(DEVICE)
            # assert all the next state is between 0 and 10
            assert (next_current_state >= 0).all() and (next_current_state <= 10).all(), f"Next state out of bounds: {next_current_state}"

            next_current_shape = batch['target_shape'].to(DEVICE)
            action = batch['action'].to(DEVICE)

            optimizer.zero_grad()

            # Forward pass: encode state and embed actions
            state_encoded = state_encoder(current_state, current_shape, dropout_eval=False)
            action_encoded = action_embedder(action)
            next_shape_logits, next_state_logits = transition_model(state_encoded, action_encoded)

            # Calculate the losses
            total_loss, state_loss, shape_loss, non_padded_loss = loss_fn(
                next_state_logits, next_shape_logits, next_current_state, next_current_shape
            )

            shape_losses.append(shape_loss.item())
            state_losses.append(state_loss.item())
            total_losses.append(total_loss.item())
            non_padded_losses.append(non_padded_loss.item())

            # Backward pass and optimization step
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            global_step += 1

            # Early stopping condition based on max_iter if provided
            if max_iter is not None and global_step >= max_iter:
                break
        
        action_embedder.save_weights(save_model_dir)
        state_encoder.save_weights(save_model_dir)
        transition_model.save_weights(save_model_dir)
        logger.info(f'Saved model weights to {save_model_dir}')
    
        # Print the average loss for the epoch
        avg_loss = np.mean(total_losses)
        avg_shape_loss = np.mean(shape_losses)
        avg_state_loss = np.mean(state_losses)
        avg_non_padded_loss = np.mean(non_padded_losses)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Shape Loss: {avg_shape_loss:.4f}, State Loss: {avg_state_loss:.4f}, Non-Padded Loss: {avg_non_padded_loss:.4f}")
        wandb.log({
            "world_model/train_loss": avg_loss,
            "world_model/train_shape_loss": avg_shape_loss,
            "world_model/train_state_loss": avg_state_loss,
            "world_model/train_non_padded_loss": avg_non_padded_loss,
        })

        # Save checkpoint every save_per_epochs epochs
        if (epoch + 1) % save_per_epochs == 0:
            checkpoint_path = os.path.join(save_model_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'state_encoder': state_encoder.state_dict(),
                'action_embedder': action_embedder.state_dict(),
                'transition_model': transition_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)

        # Evaluate the model at specified intervals
        if (epoch + 1) % eval_interval == 0:
            eval_loss, eval_shape_loss, eval_state_loss, eval_non_padded_loss = evaluate_world_model(
                state_encoder, action_embedder, transition_model, test_loader, DEVICE
            )
            logger.info(f"Evaluation - Loss: {eval_loss:.4f}, Shape Loss: {eval_shape_loss:.4f}, State Loss: {eval_state_loss:.4f}. Non-Padded Loss: {eval_non_padded_loss:.4f}")
            wandb.log({
                "world_model/eval_loss": eval_loss,
                "world_model/eval_shape_loss": eval_shape_loss,
                "world_model/eval_state_loss": eval_state_loss,
                "world_model/eval_non_padded_loss": eval_non_padded_loss,
            })

            # Early stopping check based on evaluation non_padded_loss
            if eval_non_padded_loss < best_eval_non_padded_loss:
                best_eval_non_padded_loss = eval_non_padded_loss
                patience_counter = 0  # Reset the counter if improvement is observed
            else:
                patience_counter += 1
                logger.info(f"No improvement in non_padded_loss for {patience_counter} consecutive epoch(s).")
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1} due to overfitting.")
                    break

        # Plot the results at specified intervals
        if (epoch+1) % plot_interval == 0:
            plot_world_model(state_encoder, action_embedder, transition_model, test_loader, DEVICE, world_model_args, epoch, save_model_dir)
            logger.info(f"Plotting results for epoch {epoch+1}")
        
        # Exit early if global_step reached max_iter
        if max_iter is not None and global_step >= max_iter:
            break

    return state_encoder, action_embedder, transition_model


def evaluate_world_model(state_encoder, action_embedder, transition_model, test_loader, device):
    """
    Evaluate the world model on the test set.

    Args:
        state_encoder (nn.Module): The state encoder network.
        action_embedder (nn.Module): The action embedder network.
        transition_model (nn.Module): The transition model.
        test_loader (DataLoader): DataLoader for the test set.
        device: Computation device (CPU or GPU).

    Returns:
        Average loss over the test set.
    """
    state_encoder.eval()
    action_embedder.eval()
    transition_model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_shape_loss = 0.0
    total_state_loss = 0.0
    total_non_padded_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            current_state = batch['current_state'].to(device)
            current_shape = batch['current_shape'].to(device)
            next_current_state = batch['target_state'].to(device)
            next_current_shape = batch['target_shape'].to(device)
            action = batch['action'].to(device)

            state_encoded = state_encoder(current_state, current_shape, dropout_eval=True)
            action_encoded = action_embedder(action)
            next_shape_logits, next_state_logits = transition_model(state_encoded, action_encoded)
            next_shape_predicted, next_state_predicted = transition_model.generate(state_encoded, action_encoded)

            """
            if i == 0:
                for j in range(2):
                    next_state_j = next_current_state[j]
                    next_shape_j = next_current_shape[j]
                    next_state_predicted_j = next_state_predicted[j]
                    next_shape_predicted_j = next_shape_predicted[j]
                    #print('next shape:', next_shape_j)
                    #print('next shape predicted:', next_shape_predicted_j)
                    #print('next state:', next_state_j.reshape(-1, 30, 30))
                    #print('next state predicted:', next_state_predicted_j.reshape(-1, 30, 30))
                    #print('overlap:', (next_state_j == next_state_predicted_j).sum())
                    #print('-' * 50)
            """
            # Calculate the loss
            batch_loss, state_loss, shape_loss, non_padded_loss = loss_fn(
                next_state_logits, next_shape_logits, next_current_state, next_current_shape
            )
            
            total_shape_loss += shape_loss.item()
            total_state_loss += state_loss.item()
            total_loss += batch_loss.item()
            total_non_padded_loss += non_padded_loss.item()
            count += 1

    avg_loss = total_loss / count if count > 0 else 0.0
    avg_shape_loss = total_shape_loss / count if count > 0 else 0.0
    avg_state_loss = total_state_loss / count if count > 0 else 0.0
    avg_non_padded_loss = total_non_padded_loss / count if count > 0 else 0.0
    return avg_loss, avg_shape_loss, avg_state_loss, avg_non_padded_loss


"""
            # Optionally perform evaluation at given intervals
            if e % eval_interval == 0:
                # Insert evaluation code here (using eval_env) if needed.
                pass

            # Optionally save model checkpoints
            if e % save_per_epochs == 0:
                model_save_path = os.path.join(save_model_dir, f"transition_model_epoch_{e}.pth")
                torch.save(transition_model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")

        # Save the final model
        state_encoder.save_model(save_model_dir)
        action_embedder.save_model(save_model_dir)
        transition_model.save_model(save_model_dir)

    return state_encoder, action_embedder, transition_model"""

def plot_world_model(state_encoder, action_embedder, transition_model,
                     test_loader, device, world_model_args, epoch, save_dir=None):
    """
    Plot the results of the world model training, saving the plots to a 'plots' folder
    within 'save_dir' if provided.
    """
    import os
    import torch

    # 1) Create a 'plots' subdirectory in save_dir
    plots_dir = None
    if save_dir is not None:
        plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)  # Make sure it exists

    state_encoder.eval()
    action_embedder.eval()
    transition_model.eval()

    batch_size = world_model_args.get('batch_size', 32)

    with torch.no_grad():
        for batch in test_loader:
            # Suppose shapes: (batch_size, 900), (batch_size, 2)
            current_state = batch['current_state'].to(device)
            current_shape = batch['current_shape'].to(device)
            next_current_state = batch['target_state'].to(device)
            next_current_shape = batch['target_shape'].to(device)
            action = batch['action'].to(device)

            # Model outputs
            state_encoded = state_encoder(current_state, current_shape, dropout_eval=True)
            action_encoded = action_embedder(action)
            next_shape, next_state = transition_model.generate(state_encoded, action_encoded)

            # Reshape
            grid_30x30_predicted = next_state.view(batch_size, 30, 30)
            grid_30x30_actual    = next_current_state.view(batch_size, 30, 30)

            # Pick 5 samples to plot
            random_indices = torch.randperm(batch_size)[:5]

            # Collect and plot data
            for i in random_indices:
                i_int = i.item()
                rows_pred, cols_pred = next_shape[i]         # shape predicted
                rows_act, cols_act = next_current_shape[i]   # shape actual

                # Convert to numpy
                predicted_full   = grid_30x30_predicted[i].cpu().numpy()
                predicted_cropped= predicted_full[:rows_pred, :cols_pred]
                actual_full      = grid_30x30_actual[i].cpu().numpy()
                actual_cropped   = actual_full[:rows_act, :cols_act]

                # 2) Build a filename in the 'plots' directory
                # e.g. "plots/predicted_full_idx42.png"
                # You can incorporate epoch, global_step, or i_int in the name
                if plots_dir is not None:
                    predicted_full_path    = os.path.join(plots_dir, f'e{epoch}_{i_int}_predicted_full.png')
                    predicted_cropped_path = os.path.join(plots_dir, f'e{epoch}_{i_int}_predicted_cropped.png')
                    actual_full_path       = os.path.join(plots_dir, f'e{epoch}_{i_int}_actual_full.png')
                    actual_cropped_path    = os.path.join(plots_dir, f'e{epoch}_{i_int}_actual_cropped.png')
                else:
                    # If no save_dir was provided, we don't save
                    predicted_full_path    = None
                    predicted_cropped_path = None
                    actual_full_path       = None
                    actual_cropped_path    = None

                # 3) Plot and save each image
                save_grid(predicted_full,
                          title=f'Predicted Full - idx={i_int}',
                          save_path=predicted_full_path)
                save_grid(predicted_cropped,
                          title=f'Predicted Cropped - idx={i_int}',
                          save_path=predicted_cropped_path)
                save_grid(actual_full,
                          title=f'Actual Full - idx={i_int}',
                          save_path=actual_full_path)
                save_grid(actual_cropped,
                          title=f'Actual Cropped - idx={i_int}',
                          save_path=actual_cropped_path)

            # Exit after plotting one batch to avoid huge spam
            break