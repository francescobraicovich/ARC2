import os
import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from utils.util import set_device
from world_model.memory_data_load import WorldModelDataset

# Determine the device: CUDA -> MPS -> CPU
DEVICE = set_device('world_model/train_world_model.py')

def world_model_train(
    state_encoder,
    action_embedder,
    transition_model,
    world_model_args,
    save_model_dir,
    logger=None,
    save_per_epochs=1,
    eval_interval=1,
):
    """
    Train the world model using state encoding, action embedding, and a transition model.
    
    Args:
        state_encoder (nn.Module): Encoder network for the state.
        action_embedder (nn.Module): Embedder network for the action.
        transition_model (nn.Module): Transition model predicting next state and shape.
        world_model_args (dict): Dictionary with training hyperparameters (epochs, lr, batch_size, max_iter).
        save_model_dir (str): Directory path to save model checkpoints.
        logger: (optional) Logger object with a log() method.
        save_per_epochs (int): Save model checkpoint every N epochs.
        eval_interval (int): Evaluate the model every N epochs.
    
    Returns:
        Tuple of trained (state_encoder, action_embedder, transition_model).
    """
    epochs = world_model_args.get('epochs', 10)
    lr = world_model_args.get('lr', 1e-3)
    batch_size = world_model_args.get('batch_size', 32)
    max_iter = world_model_args.get('max_iter', None)

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
        for batch in progress_bar:
            # Transfer data to the device
            current_state = batch['current_state'].to(DEVICE)
            current_shape = batch['current_shape'].to(DEVICE)
            next_current_state = batch['target_state'].to(DEVICE)
            next_current_shape = batch['target_shape'].to(DEVICE)
            action = batch['action'].to(DEVICE)

            optimizer.zero_grad()

            # Forward pass: encode state and embed actions
            state_encoded = state_encoder(current_state, current_shape, dropout_eval=False)
            action_encoded = action_embedder(action)
            next_shape_logits, next_state_logits = transition_model(state_encoded, action_encoded)

            shape_logits = next_shape_logits.permute(0, 2, 1)
            state_logits = next_state_logits.permute(0, 2, 1)

            # Compute individual losses.
            shape_loss = criterion(shape_logits, next_current_shape)
            state_loss = criterion(state_logits, next_current_state)

            # Compute the total loss with weighting.
            total_loss = shape_loss + state_loss

            shape_losses.append(shape_loss.item())
            state_losses.append(state_loss.item())
            total_losses.append(total_loss.item())

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
    
        # print the average loss
        avg_loss = np.mean(total_losses)
        avg_shape_loss = np.mean(shape_losses)
        avg_state_loss = np.mean(state_losses)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Shape Loss: {avg_shape_loss:.4f}, State Loss: {avg_state_loss:.4f}")
        wandb.log({
            "train_loss": avg_loss,
            "train_shape_loss": avg_shape_loss,
            "train_state_loss": avg_state_loss,
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
            eval_loss, eval_shape_loss, eval_state_loss = evaluate_world_model(
                state_encoder, action_embedder, transition_model, test_loader, DEVICE
            )
            logger.info(f"Evaluation - Loss: {eval_loss:.4f}, Shape Loss: {eval_shape_loss:.4f}, State Loss: {eval_state_loss:.4f}")
            wandb.log({
                "eval_loss": eval_loss,
                "eval_shape_loss": eval_shape_loss,
                "eval_state_loss": eval_state_loss,
            })

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
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            current_state = batch['current_state'].to(device)
            current_shape = batch['current_shape'].to(device)
            next_current_state = batch['target_state'].to(device)
            next_current_shape = batch['target_shape'].to(device)
            action = batch['action'].to(device)

            state_encoded = state_encoder(current_state, current_shape, dropout_eval=True)
            action_encoded = action_embedder(action)
            next_shape_logits, next_state_logits = transition_model(state_encoded, action_encoded)

            shape_logits = next_shape_logits.permute(0, 2, 1)
            state_logits = next_state_logits.permute(0, 2, 1)

            shape_loss = criterion(shape_logits, next_current_shape)
            state_loss = criterion(state_logits, next_current_state)
            batch_loss = shape_loss + state_loss


            total_shape_loss += shape_loss.item()
            total_state_loss += state_loss.item()
            total_loss += batch_loss.item()
            count += 1

    avg_loss = total_loss / count if count > 0 else 0.0
    avg_shape_loss = total_shape_loss / count if count > 0 else 0.0
    avg_state_loss = total_state_loss / count if count > 0 else 0.0
    return avg_loss, avg_shape_loss, avg_state_loss


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