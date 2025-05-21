# ReinforcementLearning

This project implements a TD3 (Twin Delayed Deep Deterministic Policy Gradient) agent for the Bipedal Walker environment using the [rldurham gym](https://github.com/XinJingHao/TD3-BipedalWalkerHardcore-v2). It extends the original work by Jinghao and adapts it for both the standard ("softcore") and "hardcore" versions of the environment.

## Features

- TD3 agent implementation in PyTorch
- Training and evaluation scripts for both softcore and hardcore environments
- Logging, statistics tracking, and video recording of agent performance
- Replay buffer for experience replay
- Demo with a predefined heuristic for comparison

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- [rldurham](https://pypi.org/project/rldurham/)
- (Optional) SWIG for some dependencies

### Installation

Install the required packages:
```bash
pip install torch rldurham swig
```

### Usage

1. **Training the Agent**

   - For the softcore environment, use the notebook:  
     `nkfn77-agent-code-softcore.ipynb`
   - For the hardcore environment, use the notebook:  
     `nkfn77-agent-code-hardcore.ipynb`

   Open the desired notebook in VS Code or Jupyter and run the cells to train the agent. Training progress, logs, and videos will be saved in the respective folders.

2. **Demo**

   Both notebooks include a demo section that runs a predefined heuristic for comparison.

3. **Logs and Videos**

   - Training logs are saved in the `logs` folder.
   - Videos of agent performance are saved in the `videos` folder.

## Credit

This project is based on the work by Jinghao:  
https://github.com/XinJingHao/TD3-BipedalWalkerHardcore-v2

Please refer to the original repository for more details and background.

---
