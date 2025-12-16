# Drift-Adaptive PPO: Robust Reinforcement Learning in Non-Stationary Environments

![Status](https://img.shields.io/badge/Status-Active_Research-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Framework](https://img.shields.io/badge/SB3-PyTorch-orange)

## 📖 Abstract

Standard Reinforcement Learning algorithms (like PPO) assume a **Stationary Markov Decision Process (MDP)**, where environment dynamics (gravity, friction, engine power) remain constant throughout training. However, real-world applications often face **Concept Drift** or non-stationary dynamics.

This repository implements a **Drift-Adaptive Scheduling mechanism** for Proximal Policy Optimization (PPO). By monitoring environmental parameters (via an Oracle or Proxy), the agent dynamically adjusts its hyperparameters (Learning Rate) to "follow the drift"—accelerating learning when dynamics change and stabilizing when the environment is static.

## 🚀 Key Features

- **Drift-Adaptive Callback:** A custom Stable-Baselines3 callback that modifies the Learning Rate schedule in real-time based on drift magnitude.
- **Universal Morphing Wrapper:** A gym wrapper designed to inject dynamic changes (Drift) into various environments (CartPole, LunarLander, HalfCheetah) and expose ground-truth parameters for analysis.
- **Experiment Tracking:** Fully integrated with **Weights & Biases (WandB)** for monitoring Policy Loss, Value Estimates, and Adaptation Factors.

## 🧠 Methodology

### The Adaptive Heuristic
The core hypothesis is derived from the "Follow the Drift" principle:
> *When the MDP moves, the agent must take larger update steps to adapt. When the MDP is stable, the agent should decay the step size for convergence.*

We implement a dynamic scaling rule for the Learning Rate ($\eta$):

$$\eta_t = \eta_{base} \times (1 + \lambda \cdot |\Delta_{env}|)$$

Where:
- $\eta_{base}$: The base learning rate (e.g., 3e-4).
- $\Delta_{env}$: The magnitude of environmental drift (e.g., $|\text{current\_gravity} - \text{base\_gravity}|$).
- $\lambda$: Sensitivity hyperparameter (Scale Factor).

## 🧪 Environments & Experiments

We are testing the robustness of this approach across varying levels of complexity:

| Complexity | Environment | Drift Parameter | Scenario | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Toy)** | `CartPole-v1` | Gravity ($g$) | Sudden changes in gravity (Jump Drift) | ✅ Completed |
| **Tier 2 (Control)** | `LunarLanderContinuous-v2` | Wind / Engine Power | Turbulence & Engine Failure | 🚧 In Progress |
| **Tier 3 (Physics)** | `HalfCheetah-v4` (MuJoCo) | Friction / Torso Mass | Slippery floor & Payload variation | 📅 Planned |

## 📂 Project Structure

```bash
.
├── callbacks/
│   └── drift_callback.py    # The core adaptive logic (SB3 Callback)
├── envs/
│   └── morphing_wrapper.py  # Wrapper to inject drift into Gym envs
├── configs/
│   └── experiment_config.yaml
├── train.py                 # Main entry point for training
└── README.md
```

## 🛠️ Usage
###1. Installation
```bash
pip install stable-baselines3 wandb gym[box2d] mujoco
```

###2. Running an ExperimentTo run a baseline (static) vs. adaptive experiment on CartPole:

```bash
python train.py --env CartPole-v1 --adaptive True --drift_type jump
```

##📊 Preliminary Results*Early experiments on CartPole-v1 show that while high-magnitude drift destabilizes standard PPO, the Adaptive mechanism helps maintain Value Function estimation accuracy (Explained Variance), though hyperparameter tuning for the Scaling Factor (\lambda) is critical to prevent catastrophic forgetting.*

##📝 To-Do List* [x] Implement basic `DriftAdaptiveCallback` for PPO.
* [x] Fix scheduler interaction bug in SB3.
* [ ] Implement **Universal Morphing Wrapper** for Box2D and MuJoCo.
* [ ] Run benchmark on `LunarLanderContinuous-v2` (Wind Drift).
* [ ] Ablation study: Compare "Increase LR on Drift" vs "Decrease LR on Uncertainty".

##🤝 ContributionThis is a research project. Feedback on the adaptive scaling logic or suggestions for more realistic drift scenarios are welcome.

```