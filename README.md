# Solving ARC with Reinforcement Learning
### A novel approach for sequential solutions


## Overview
This repository offers a reinforcement learning framework designed to tackle tasks from the Abstraction and Reasoning Corpus (ARC). It includes a Domain-Specific Language (DSL) for solving ARC problems sequentially and a custom RL environment. The implemented model is a Wolpertinger Actor-Critic, featuring a choice of feature extractors—either LPN or CNN, both of which are provided.

## Paper
<iframe src="ARC-Hephaestus.pdf" width="100%" height="600px"></iframe>

## Installation
1. Clone this repository.  
2. Install dependencies from the `requirements.txt.
    ```bash
    pip install -r requirements.txt
    ```

## Usage
- Run training with:
  ```bash
  python src/main.py
  ```
- Adjust hyperparameters in `src/arg_parser.py`.

## Folder Structure
- **src/**: Core source code including agents, environment, and utilities.  
- **data/**: Contains ARC challenges in JSON format.  
- **src/dsl/**: Contains the Domain Specific Language (DSL) to solve ARC problems.
- **src/rearc/**: Contains code to generate new ARC problems (REARC).
- **src/embedded_space/**: Contains code to generate the action space embedding.
- **README.md** (this file).

## License
Distributed under the MIT License.

