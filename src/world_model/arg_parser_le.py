import argparse

def init_parser_le():
    parser = argparse.ArgumentParser(description="Learnable Embedding Training Arguments")
    # Training loop hyperparameters
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of pretraining epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (e.g., cuda or cpu)")
    parser.add_argument("--noise-level", type=float, default=0.0, help="Noise level for data augmentation")
    parser.add_argument("--max-episode-length", type=int, default=30, help="Max steps per episode")
    parser.add_argument("--max-episode", type=int, default=500000, help="Maximum number of episodes")
    parser.add_argument("--max-actions", type=int, default=int(1e9), help="Maximum number of actions")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of training steps")
    
    # Model hyperparameters (referencing model_le.py defaults)
    parser.add_argument("--num-actions", type=int, default=50000, help="Total number of discrete actions")
    parser.add_argument("--action-embed-dim", type=int, default=32, help="Dimension for action embeddings")
    parser.add_argument("--state-embed-dim", type=int, default=64, help="Dimension for state embeddings")
    parser.add_argument("--diff-in-channels", type=int, default=3, help="Input channels for UNet")
    parser.add_argument("--diff-out-channels", type=int, default=3, help="Output channels for UNet")
    parser.add_argument("--observation-shape", type=int, nargs=3, default=[224,224,3],
                        help="Observation shape (H W C), used by the model")
    return parser

if __name__ == "__main__":
    # For testing the argparser individually.
    parser = init_parser_le()
    args = parser.parse_args()
    print(args)
