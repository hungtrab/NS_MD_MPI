# Implementation Guide: Regularized Non-Stationary MDPs (NS-MD-MPI)

## 1. Project Objective
We are implementing the **Drift-Adaptive NS-MD-MPI** algorithm from the paper "Regularized Non-Stationary Markov Decision Processes". The goal is to solve Reinforcement Learning tasks where rewards and transition dynamics drift over time (Non-Stationary MDPs).

**Key Differentiator:** unlike standard RL, we must track the environment's drift ($\hat{\Delta}_t$) and automatically adapt the trust-region radius ($\delta_t$) or entropy temperature ($\eta_t$) in real-time.

---

## 2. Core Algorithm: NS-MD-MPI (Algorithm 1)

The training loop proceeds as follows for each time step $t$:

1.  **Collect Data:** Observe transition $(s, a, r, s')$ and store in buffer.
2.  **Estimate Drift:** Compute proxies $\hat{\Delta}_t^R$ (reward drift) and $\hat{\Delta}_t^P$ (transition drift).
3.  **Adapt Hyperparameters:** Calculate dynamic $\delta_t$ or $\eta_t$ based on drift.
4.  **Greedy Step (Policy Update):** Update policy $\pi_{t+1}$ using Mirror Descent.
5.  **Evaluation Step (Value Update):** Update $Q$-values using $m$-step lookahead.

### The Adaptive Schedule (Crucial)
Do not use fixed hyperparameters. Use the **Drift-Adaptive Schedule** (Sec 7.3):

* **Trust Region Form:** $\delta_t = \text{clip}(\delta_{\min}, \delta_{\max}, c_0 + c_1 \cdot \text{EMA}_\tau(\hat{\Delta}_t))$
* **Temperature Form:** $\eta_t = \text{clip}(\eta_{\min}, \eta_{\max}, d_0 + d_1 \cdot \text{EMA}_\tau(\hat{\Delta}_t))$

Where:
* $\hat{\Delta}_t = \hat{\Delta}_t^R + 2\gamma B \hat{\Delta}_t^P$ (The Bellman Commutator proxy).
* $\text{EMA}_\tau$ is the Exponential Moving Average with window $\tau$.

---

## 3. Drift Estimation Implementation (Appendix E)

We cannot know the true $P_t$ and $R_t$. We must estimate them using samples from the current batch ($t$) vs the previous batch ($t-1$).

### A. Reward Drift ($\hat{\Delta}_t^R$)
* **Method:** Empirical mean difference.
* **Formula:** $\hat{\Delta}_t^R = \max_{s,a} |\hat{\mu}_t(s,a) - \hat{\mu}_{t-1}(s,a)|$.
* **For Continuous Rewards:** Use Huber loss difference or simply $|r_t - r_{t-1}|$ averaged.

### B. Transition Drift ($\hat{\Delta}_t^P$) - The Hard Part
We need the Total Variation (TV) distance between $P_t(\cdot|s,a)$ and $P_{t-1}(\cdot|s,a)$.

* **Discrete State Space:** Use L1 distance of normalized histograms.
* **Continuous State Space (e.g., MuJoCo):** Use the **Classifier-Based Proxy** (Appendix E.3).
    1.  Create a dataset of transitions labeled `0` for time $t-1$ and `1` for time $t$.
    2.  Train a lightweight binary classifier (e.g., small MLP) to discriminate $(s, a, s')$.
    3.  Compute accuracy on a held-out set.
    4.  **Proxy Formula:** $\hat{TV} \approx 2 \times (\text{Accuracy} - 0.5)$.

---

## 4. Policy Optimization (The Greedy Step)

We use **Mirror Descent** (Sec 5). The update rule maximizes the Q-value minus a Bregman divergence penalty.

* **Objective:** $\pi_{t}^{(k+1)} = \arg\max_{\pi} \langle Q_t, \pi \rangle - \frac{1}{\eta_t} D_{\Omega}(\pi || \pi_{t}^{(k)})$
* **Practical Implementation:**
    * If $\Omega$ is Shannon Entropy (standard RL), this becomes a **Softmax update** or standard **KL-regularized update** (like PPO/TRPO) but with the **dynamic coefficient** $\eta_t$.
    * **Do not** use fixed clip ranges (PPO) or fixed alphas (SAC). Inject the adaptive $\delta_t$ or $\eta_t$ calculated in Section 2.

---

## 5. Experiment Setup

### A. Environment & Drift Generators
We need to modify standard environments (GridWorld, CartPole, HalfCheetah) to be non-stationary.
* **Drift Types:**
    1.  **Jumps:** Parameters change abruptly every $N$ steps.
    2.  **Ramps:** Parameters drift linearly (e.g., gravity increases by 0.01 every step).
    3.  **Sinusoid:** Parameters oscillate over time.
* **Target Parameters:** Reward magnitudes, Friction, Gravity, Mass, Wind.

### B. Evaluation Metrics (Algorithm 2)
1.  **Dynamic Regret:** $DynReg(T) = \sum_{t=1}^T (V_t^* - V_t^{\pi_t})$.
    * *Note:* To get $V_t^*$ (Oracle), we must "freeze" the environment at step $t$ and train a separate expert policy to convergence.
2.  **Drift Tracking:** Plot $\hat{\Delta}_t$ (estimated) vs. True Parameter Drift to verify the estimator.

---

## 6. Summary of Implementation Priorities
1.  **Drift Estimator Module:** Implement the Classifier-based TV proxy first.
2.  **Adaptive Scheduler:** Create a class that takes drift history and outputs $\eta_t$.
3.  **Agent Loop:** Modify a standard actor-critic loop (like SAC or PPO) to accept the dynamic $\eta_t$ per step instead of a fixed arg.