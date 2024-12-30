import torch
import numpy as np
import datetime
from action_space import ARCActionSpace

device = torch.device("cpu")
print(f"Using device: {device}")

action_space = ARCActionSpace()
actions = action_space.get_space()
n = len(actions)

def apply_action(grid, action):
    # If everything is on CPU/NumPy, adapt as needed
    color_key, selection_key, transform_key = action
    color_func = action_space.color_selection_dict[color_key]
    selection_func = action_space.selection_dict[selection_key]
    transform_func = action_space.transformation_dict[transform_key]

    grid_np = grid.numpy().astype(np.int64)
    color = color_func(grid_np)
    selected = selection_func(grid_np, color=color)

    if selected.any():
        result = transform_func(grid_np, selection=selected)
        return torch.from_numpy(result)
    else:
        return grid.unsqueeze(0)


def build_action_action_similarity_matrix_experiments(num_experiments=1, alpha=0.5):
    # We'll store similarity here
    similarity_matrix = torch.zeros((n, n), dtype=torch.float32)

    for experiment in range(num_experiments):
        # 1) Generate random input
        rand_array = torch.randint(0, 9, (30, 30), dtype=torch.int8)

        # 2) Apply each action -> outputs list
        outputs = [apply_action(rand_array, action) for action in actions]

        # 3) Convert each output to shape (30, 30) if needed
        def ensure_2d(t):
            return t.squeeze(0) if t.ndim == 3 else t
        outputs_pt = [ensure_2d(o) for o in outputs]

        # 4) Stack them: shape [n, 30, 30], convert to float
        out_stack = torch.stack(outputs_pt, dim=0).float()

        # Preallocate buffers for intersection and union
        intersection = torch.empty_like(out_stack)
        union = torch.empty_like(out_stack)

        # 5) Compute row-by-row
        for i in range(n):
            # Calculate intersection and union in-place
            torch.min(out_stack[i].unsqueeze(0), out_stack, out=intersection)
            torch.max(out_stack[i].unsqueeze(0), out_stack, out=union)

            intersection_sum = intersection.sum(dim=(-1, -2))  # shape [n]
            union_sum = union.sum(dim=(-1, -2))                # shape [n]

            # Compute overlap in-place
            overlap = intersection_sum.div_(union_sum).mul_(100.0)  # shape [n]

            # alpha-blend with existing similarity
            similarity_matrix[i].mul_(1.0 - alpha).add_(alpha * overlap)

        print(f"Completed experiment {experiment + 1}/{num_experiments}")

    return similarity_matrix
def save_similarity_matrix_np(s_matrix, base_filename="similarity_matrix"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.npy"
    np.save(filename, s_matrix.numpy())
    print(f"Similarity matrix saved as {filename}")

if __name__ == "__main__":
    similarity_matrix = build_action_action_similarity_matrix_experiments(num_experiments=100, alpha=0.7)
    print(similarity_matrix)
    save_similarity_matrix_np(similarity_matrix)