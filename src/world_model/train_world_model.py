import os
import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import torch

from utils.util import set_device
from world_model.memory_data_load import IterableWorldModelDataset

from dsl.utilities.plot import save_grids, to_numpy

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
    logger=None,      # Assuming a logger object is passed
    save_per_epochs=1,
    eval_interval=1,
    plot_interval=1
):
    """
    Train the world model using the IterableWorldModelDataset.
    """
    # --- Argument Extraction ---
    epochs = world_model_args.get('epochs', 10)
    lr = world_model_args.get('lr', 1e-3)
    batch_size = world_model_args.get('batch_size', 32)
    max_iter = world_model_args.get('max_iter', None)
    patience = world_model_args.get('early_stopping_patience', 5)
    
    # **MODIFICATION 1:** Get dataset paths from args
    memory_chunk_dir = world_model_args.get('load_memory_dir')
    if not memory_chunk_dir:
        raise ValueError("world_model_args must contain 'memory_chunk_dir' and 'evaluation_file_path'")


    best_eval_non_padded_loss = float('inf')
    patience_counter = 0

    # Create optimizer (no change needed here)
    optimizer = torch.optim.Adam(
        list(state_encoder.parameters()) +
        list(action_embedder.parameters()) +
        list(transition_model.parameters()),
        lr=lr
    )

    # Loss function (assuming nn.CrossEntropyLoss or similar is used within loss_fn)
    # criterion = nn.CrossEntropyLoss() # Keep if used directly, otherwise handled by loss_fn

    # --- Dataset Initialization and DataLoader Creation ---
    # We assume chunk shuffling is desired for training randomness
    dataset = IterableWorldModelDataset(
        memory_chunk_dir=memory_chunk_dir,
        chunk_shuffle=True
    )

    train_set = dataset # The iterable dataset is the training set
    test_set = dataset.get_evaluation_dataset() # Get the map-style evaluation set

    if test_set is None:
        # This should have been caught by dataset init, but double-check
        raise RuntimeError("Evaluation dataset failed to load. Cannot proceed.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False, # Shuffling is handled by chunk_shuffle in the dataset iterator
        num_workers=world_model_args.get('num_workers', 0), # Allow configuring num_workers
        pin_memory=world_model_args.get('pin_memory', False) # Allow configuring pin_memory
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size, # Can use a different batch size for evaluation if needed
        shuffle=True # Shuffle the map-style evaluation dataset
    )
    # --- End Dataset Initialization ---

    try:
        test_length = len(test_set) # Length is available for the map-style test_set
        if test_length == 0:
             print("Warning: Evaluation dataset is empty.")
             random_indices = []
        else:
             plot_sample_size = min(15, test_length) # Ensure we don't request more samples than available
             random_indices = np.random.choice(test_length, size=plot_sample_size, replace=False)
    except TypeError:
        # Should not happen if test_set is _MapStyleEvaluationDataset, but defensive check
        print("Warning: Could not determine length of evaluation dataset. Plotting indices set to empty.")
        test_length = 0
        random_indices = []

    # Parameter counting (no change needed)
    action_embedding_num_params = sum(p.numel() for p in action_embedder.parameters()) # Example if needed
    state_encoder_num_params = sum(p.numel() for p in state_encoder.parameters())     # Example if needed
    transition_model_num_params = sum(p.numel() for p in transition_model.parameters()) # Example if needed

    # Logging setup (assuming logger exists)
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO) # Basic config if none provided

    print('-' * 50)
    logger.info('Starting World Model Training with Iterable Dataset')
    logger.info(f"Action Embedding Parameters: {action_embedding_num_params}")
    logger.info(f"State Encoder Parameters: {state_encoder_num_params}")
    logger.info(f"Transition Model Parameters: {transition_model_num_params}")
    logger.info(f"Training data source: Directory {memory_chunk_dir}")
    logger.info(f"Device: {DEVICE}")


    global_step = 0
    for epoch in range(epochs):
        state_encoder.train()
        action_embedder.train()
        transition_model.train()
        
        running_loss = 0.0
        # **MODIFICATION 6:** TQDM might not show total length for iterable loader
        # It will show progress based on batches processed.
        files_processed_str = "0/0" # Initialize display string
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        # Reset epoch-specific metrics
        total_losses = []
        shape_losses = []
        state_losses = []
        non_padded_losses = []

        # --- Training Loop ---
        for batch in progress_bar:

            # But keeping .to(DEVICE) here is harmless and explicit.
            current_state = batch['current_state'].to(DEVICE)
            current_shape = batch['current_shape'].to(DEVICE)
            next_current_state = batch['target_state'].to(DEVICE)
            next_current_shape = batch['target_shape'].to(DEVICE)
            action = batch['action'].to(DEVICE)

            # Assertion remains valid
            assert (next_current_state >= 0).all() and (next_current_state <= 10).all(), f"Next state out of bounds: {next_current_state}"

            optimizer.zero_grad()

            # Forward pass (no change needed)
            state_encoded = state_encoder(current_state, current_shape, dropout_eval=False)
            action_encoded = action_embedder(action)
            next_shape_logits, next_state_logits = transition_model(state_encoded, action_encoded)

            # Calculate the losses (assuming loss_fn signature matches)
            total_loss, state_loss, shape_loss, non_padded_loss = loss_fn(
                next_state_logits, next_shape_logits, next_current_state, next_current_shape
            )

            # Accumulate losses
            shape_losses.append(shape_loss.item())
            state_losses.append(state_loss.item())
            total_losses.append(total_loss.item())
            non_padded_losses.append(non_padded_loss.item())

            # Backward pass and optimization step (no change needed)
            total_loss.backward()
            # Optional: Gradient clipping can be added here if needed
            # torch.nn.utils.clip_grad_norm_(...)
            optimizer.step()

            running_loss += total_loss.item() # Keep for potential intermediate logging
            global_step += 1
            
            current_idx = train_set.current_file_index 
            total_f = train_set.total_files
            # Clamp index to be at least 0 for display before first file loads
            files_processed_str = f"{max(0, current_idx)+1}/{total_f}" 

            # Combine file progress with loss (use last batch's loss)
            progress_bar.set_postfix_str(f"Files: {files_processed_str}, Loss: {total_loss:.4f}")

            # Early stopping condition based on max_iter (no change needed)
            if max_iter is not None and global_step >= max_iter:
                logger.info(f"Reached max_iter ({max_iter}). Stopping training.")
                break
        # --- End Training Loop for Epoch ---
        progress_bar.close() # Close the tqdm bar for the epoch

        # --- Post-Epoch Actions ---
        # Saving weights (consider saving based on best eval loss instead of every epoch)
        # The current logic saves at the end of each epoch's training loop.
        try:
            # Assuming these save methods exist
            action_embedder.save_weights(save_model_dir)
            state_encoder.save_weights(save_model_dir)
            transition_model.save_weights(save_model_dir)
            logger.info(f'Saved latest model weights to {save_model_dir}')
        except AttributeError as e:
             logger.warning(f"Could not save weights via specific methods: {e}. Consider saving state_dict.")
        except Exception as e:
             logger.error(f"Error saving model weights: {e}")

        # Calculate and log average losses for the epoch
        # Check if any batches were processed
        if not total_losses:
             logger.warning(f"Epoch {epoch+1}/{epochs} - No batches processed. Skipping loss logging and evaluation.")
             continue # Skip to next epoch if training loop didn't run (e.g., max_iter=0)

        avg_loss = np.mean(total_losses)
        avg_shape_loss = np.mean(shape_losses)
        avg_state_loss = np.mean(state_losses)
        avg_non_padded_loss = np.mean(non_padded_losses)
        logger.info(f"Epoch {epoch+1}/{epochs} Train Summary - Avg Loss: {avg_loss:.4f}, Shape: {avg_shape_loss:.4f}, State: {avg_state_loss:.4f}, NonPad: {avg_non_padded_loss:.4f}")
        
        # Log to wandb (no change needed in logging logic itself)
        if wandb.run is not None: # Check if wandb is active
            wandb.log({
                "epoch": epoch + 1,
                "global_step": global_step,
                "world_model/train_loss": avg_loss,
                "world_model/train_shape_loss": avg_shape_loss,
                "world_model/train_state_loss": avg_state_loss,
                "world_model/train_non_padded_loss": avg_non_padded_loss,
            })

        # Save checkpoint every save_per_epochs epochs (no change needed)
        if (epoch + 1) % save_per_epochs == 0:
            checkpoint_path = os.path.join(save_model_dir, f'checkpoint_epoch_{epoch+1}.pt')
            try:
                 torch.save({
                     'state_encoder': state_encoder.state_dict(),
                     'action_embedder': action_embedder.state_dict(),
                     'transition_model': transition_model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch,
                     'global_step': global_step,
                     'best_eval_loss': best_eval_non_padded_loss # Save best loss for resuming
                 }, checkpoint_path)
                 logger.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                 logger.error(f"Error saving checkpoint: {e}")

        # Evaluate the model at specified intervals (no change needed in call)
        if (epoch + 1) % eval_interval == 0 and len(test_set) > 0: # Avoid eval if test set is empty
            eval_loss, eval_shape_loss, eval_state_loss, eval_non_padded_loss = evaluate_world_model(
                state_encoder, action_embedder, transition_model, test_loader, DEVICE
            )
            logger.info(f"Epoch {epoch+1}/{epochs} Evaluation - Loss: {eval_loss:.4f}, Shape: {eval_shape_loss:.4f}, State: {eval_state_loss:.4f}, NonPad: {eval_non_padded_loss:.4f}")
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "world_model/eval_loss": eval_loss,
                    "world_model/eval_shape_loss": eval_shape_loss,
                    "world_model/eval_state_loss": eval_state_loss,
                    "world_model/eval_non_padded_loss": eval_non_padded_loss,
                })

            # Early stopping check (no change needed)
            if eval_non_padded_loss < best_eval_non_padded_loss:
                best_eval_non_padded_loss = eval_non_padded_loss
                patience_counter = 0
                # Optionally save the best model weights here
                best_model_path = os.path.join(save_model_dir, 'best_model.pt')
                try:
                    torch.save({
                        'state_encoder': state_encoder.state_dict(),
                        'action_embedder': action_embedder.state_dict(),
                        'transition_model': transition_model.state_dict(),
                        'epoch': epoch,
                        'loss': best_eval_non_padded_loss
                    }, best_model_path)
                    logger.info(f"Saved new best model with eval loss {best_eval_non_padded_loss:.4f} to {best_model_path}")
                except Exception as e:
                     logger.error(f"Error saving best model: {e}")

            else:
                patience_counter += 1
                logger.info(f"No improvement in non_padded_loss for {patience_counter} consecutive evaluation(s).")
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1} due to lack of improvement.")
                    break # Exit epoch loop

        # Plot the results at specified intervals (no change needed in call, uses test_loader)
        if (epoch+1) % plot_interval == 0 and len(random_indices) > 0: # Check if indices exist
            try:
                plot_world_model(state_encoder = state_encoder,
                                action_embedder = action_embedder,
                                transition_model = transition_model,
                                test_loader = test_loader, # Uses the correct loader
                                device = DEVICE,
                                epoch = epoch+1,
                                selected_indices=random_indices, # Uses indices from test_set
                                save_dir=save_model_dir)
                logger.info(f"Plotting results for epoch {epoch+1}")
            except Exception as e:
                 logger.error(f"Error during plotting: {e}")
        
        # Exit early if global_step reached max_iter (check again after epoch finished)
        if max_iter is not None and global_step >= max_iter:
            break # Exit epoch loop
    # --- End Epoch Loop ---

    logger.info("World Model training finished.")
    # Optional: Load best model weights before returning if early stopping occurred
    best_model_path = os.path.join(save_model_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
         logger.info(f"Loading best model weights from {best_model_path}")
         try:
             checkpoint = torch.load(best_model_path, map_location=DEVICE)
             state_encoder.load_state_dict(checkpoint['state_encoder'])
             action_embedder.load_state_dict(checkpoint['action_embedder'])
             transition_model.load_state_dict(checkpoint['transition_model'])
         except Exception as e:
             logger.error(f"Error loading best model weights: {e}")


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
                     test_loader, device, epoch, selected_indices, save_dir=None):
    """
    Plot the world model predictions versus the actual states for specified indices.
    
    This function:
      - Constructs a sub-batch directly from test_loader.dataset using the provided numpy array of selected indices.
      - Generates predictions for this sub-batch.
      - For each sample in the sub-batch, it uses the 'save_grids' function to create a combined plot 
        that shows both the actual full and predicted full state images.
      - The title and file path for each plot include the epoch and the corresponding sample index for clarity.
      - Saves the plots in a 'plots' subdirectory within save_dir (if provided).
    
    Parameters:
      state_encoder: The model that encodes the current state.
      action_embedder: The model that embeds the action.
      transition_model: The model that generates the next state.
      test_loader: A DataLoader whose dataset supports indexing (e.g., a dict-style dataset with keys such as 
                   'current_state', 'current_shape', 'target_state', 'target_shape', and 'action').
      device: The device (CPU or GPU) to perform computations.
      world_model_args: A dict containing world model settings (e.g. batch_size).
      epoch: Current training epoch (used for naming plots).
      selected_indices: A numpy array of indices to extract from the dataset.
      save_dir: Optional base directory to save plots. If provided, plots will be saved under a 'plots' subdirectory.
    """

    # Create 'plots' subdirectory if a save directory is provided
    plots_dir = None
    if save_dir is not None:
        plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    # Set models to evaluation mode
    state_encoder.eval()
    action_embedder.eval()
    transition_model.eval()

    # Construct a batch from the dataset using the selected indices
    dataset = test_loader.dataset
    # Assuming the dataset returns a dictionary for each sample
    batch = {}
    sample_keys = dataset[0].keys()
    for key in sample_keys:
        # Gather the tensors for each selected index and stack them along the batch dimension
        batch[key] = torch.stack([dataset[i][key] for i in selected_indices], dim=0).to(device)

    # Extract the relevant tensors from the batch
    current_state      = batch['current_state']
    current_shape      = batch['current_shape']
    next_current_state = batch['target_state']
    next_current_shape = batch['target_shape']
    action             = batch['action']

    with torch.no_grad():
        # Generate model outputs
        state_encoded  = state_encoder(current_state, current_shape, dropout_eval=True)
        action_encoded = action_embedder(action)
        next_shape, next_state = transition_model.generate(state_encoded, action_encoded)

        # Reshape to obtain full grid images (assume each is of size 30x30)
        predicted_full = next_state.view(-1, 30, 30)
        actual_full    = next_current_state.view(-1, 30, 30)

        # For each sample in the sub-batch, plot and save the actual and predicted grids
        for i, sample_index in enumerate(selected_indices):
            # Convert the tensors to numpy arrays for the current sample
            pred = predicted_full[i].cpu().numpy()
            act  = actual_full[i].cpu().numpy()

            # Build a descriptive file name and title
            if plots_dir is not None:
                plot_filename = os.path.join(plots_dir, f'epoch_{epoch}_sample_{sample_index}.png')
            else:
                plot_filename = None
            title = f"Epoch {epoch} - Sample {sample_index}: Actual vs Predicted"

            # Call the save_grids function which takes grid1 (actual), grid2 (predicted), title, and save_path
            save_grids(act, pred, title=title, save_path=plot_filename)