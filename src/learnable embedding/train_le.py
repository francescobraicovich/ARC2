import wandb
import os
import torch
from utils.util import set_device
from errors_optimizer_le import max_overlap_loss

# Determine the device: CUDA -> MPS -> CPU
DEVICE = set_device()
print("Using device for model:", DEVICE)

def world_model_train(
        state_encoder,
        action_embedder,
        transition_model,
        world_model_args,
        memory,
        memory_dir,
        save_model_dir,
        logger,
        save_per_epochs,
        eval_interval
):

    # Load memory from disk and reassign (if needed)
    memory = memory.load_memory(memory_dir)

    # Hyperparameters from world_model_args
    epochs = world_model_args.epochs
    lr = world_model_args.lr
    batch_size = world_model_args.batch_size
    max_iter = world_model_args.max_iter

    # Create an optimizer for all network parameters
    optimizer = torch.optim.Adam(
        list(state_encoder.parameters()) +
        list(action_embedder.parameters()) +
        list(transition_model.parameters()),
        lr=lr
    )

    # Loss function for shape prediction
    mse_loss = torch.nn.MSELoss()

    for e in range(epochs):
        for i in range(max_iter):
            # Sample a batch of transitions from memory (all tensors with a batch dimension)
            (state_batch, shape_batch, action_batch,
            reward_batch, next_state_batch, next_shape_batch,
            terminal_batch) = memory.sample_and_split(batch_size)

            # Ensure the tensors are on the correct device and require gradients when needed
            state_batch = state_batch.to(DEVICE).requires_grad_()
            shape_batch = shape_batch.to(DEVICE).requires_grad_()
            action_batch = action_batch.to(DEVICE)
            next_state_batch = next_state_batch.to(DEVICE)
            next_shape_batch = next_shape_batch.to(DEVICE)

            # Forward pass: encode the current state and shape, and embed the batch of actions.
            state_encoded = state_encoder(state_batch, shape_batch)
            action_encoded = action_embedder(action_batch)

            # Concatenate the state and action embeddings along the feature dimension
            cond = torch.cat([state_encoded, action_encoded], dim=1)

            # Predict the next state and shape for the entire batch
            predicted_next_state, predicted_next_shape = transition_model(state_batch, cond)

            # Compute loss over the batch:
            # Use a custom max_overlap_loss for the state prediction and MSE for the shape prediction.
            loss_state = max_overlap_loss(next_state_batch, predicted_next_state)
            loss_shape = mse_loss(next_shape_batch, predicted_next_shape)
            total_loss = loss_state + loss_shape

            # Backpropagation: reset gradients, compute gradients, and update parameters
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Optionally log training progress
            if logger is not None:
                logger.log({
                    'epoch': e,
                    'loss_state': loss_state.item(),
                    'loss_shape': loss_shape.item(),
                    'total_loss': total_loss.item()
                })

            wandb.log({
                'epoch': e,
                'iteration': i,
                'loss_state': loss_state.item(),
                'loss_shape': loss_shape.item(),
                'total_loss': total_loss.item()
            })

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

    return state_encoder, action_embedder, transition_model

