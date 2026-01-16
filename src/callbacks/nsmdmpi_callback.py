"""
NS-MD-MPI Callback for Regularized Non-Stationary MDPs.

Implements Algorithm 1 (NS-MD-MPI) from the paper with:
- Variation budget tracking (V_R, V_P, V_π*)
- Adaptive trust region κ_t based on remaining budgets
- Adaptive regularization λ_t based on drift magnitude
- Budget-aware hyperparameter scheduling

Reference: Algorithm 1 - NS-MD-MPI with Drift-Adaptive Trust Region
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from stable_baselines3.common.callbacks import BaseCallback

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.estimators import (
    CombinedDriftEstimator,
    DriftEstimatorConfig,
)
from src.evaluation.variation_budgets import (
    VariationBudgetTracker,
    estimate_budgets_from_drift_config,
)


class NSMDMPICallback(BaseCallback):
    """
    Implements NS-MD-MPI (Algorithm 1) from the paper.
    
    Key Features:
    1. Tracks variation budgets V_R, V_P, V_π* over training
    2. Computes adaptive trust region κ_t based on remaining budgets
    3. Applies budget-aware regularization via hyperparameter adaptation
    4. Uses drift proxies Δ̂_R,t, Δ̂_P,t, Δ̂_C,t for policy updates
    
    Trust Region Adaptation:
        κ_t = κ_0 / (1 + α * (1 - min_budget_frac))
        
        When budgets are nearly exhausted (min_budget_frac → 0):
            - Trust region shrinks (κ_t → 0)
            - More conservative policy updates
            - Prevents overfitting to transient environment states
    
    Regularization Adaptation:
        λ_t = λ_0 * (1 + β * combined_drift)
        
        When drift increases:
            - Stronger regularization (larger λ_t)
            - Penalizes large policy changes
            - Smooths learning in non-stationary environments
    
    Implementation via Hyperparameters:
        - PPO: Trust region via clip_range = κ_t
        - TRPO: Trust region via target_kl = κ_t
        - All: Regularization via entropy coefficient and learning rate
    """
    
    def __init__(
        self,
        # Variation budgets
        V_R: float = 10.0,
        V_P: float = 10.0,
        V_pi_star: float = 5.0,
        auto_estimate_budgets: bool = False,
        budget_scale_factor: float = 1.5,
        
        # Trust region
        kappa_base: float = 0.2,
        kappa_min: float = 0.05,
        kappa_max: float = 0.4,
        kappa_adaptive: bool = True,
        trust_region_sensitivity: float = 5.0,  # α in formula
        
        # Regularization
        lambda_base: float = 1.0,
        lambda_min: float = 0.1,
        lambda_max: float = 10.0,
        lambda_adaptive: bool = True,
        regularization_sensitivity: float = 2.0,  # β in formula
        
        # Drift estimation
        drift_weights: Tuple[float, float, float] = (1.0, 1.0, 0.5),
        drift_window_size: int = 1000,
        drift_min_samples: int = 100,
        
        # Entropy adaptation (for exploration)
        adapt_entropy: bool = True,
        base_ent_coef: float = 0.0,  # 0 = auto-detect
        min_ent_coef: float = 0.0,
        max_ent_coef: float = 0.1,
        
        # Logging
        log_freq: int = 100,
        save_budget_history: bool = True,
        budget_save_dir: str = "budgets/",
        
        verbose: int = 1,
    ):
        """
        Initialize NS-MD-MPI callback.
        
        Args:
            V_R: Reward variation budget
            V_P: Transition variation budget
            V_pi_star: Optimal policy variation budget
            auto_estimate_budgets: Auto-estimate from drift config if available
            budget_scale_factor: Scale factor for auto-estimation (>1 = more conservative)
            
            kappa_base: Base trust region size (κ_0)
            kappa_min: Minimum trust region size
            kappa_max: Maximum trust region size
            kappa_adaptive: Whether to adapt trust region based on budgets
            trust_region_sensitivity: How aggressively to shrink trust region (α)
            
            lambda_base: Base regularization coefficient (λ_0)
            lambda_min: Minimum regularization
            lambda_max: Maximum regularization
            lambda_adaptive: Whether to adapt regularization based on drift
            regularization_sensitivity: How aggressively to increase regularization (β)
            
            drift_weights: Weights for (Δ_R, Δ_P, Δ_C) in combined drift
            drift_window_size: Window size for drift estimation
            drift_min_samples: Minimum samples before drift estimation
            
            adapt_entropy: Whether to adapt entropy coefficient
            base_ent_coef: Base entropy coefficient (0 = auto-detect from model)
            min_ent_coef: Minimum entropy coefficient
            max_ent_coef: Maximum entropy coefficient
            
            log_freq: Logging frequency (in steps)
            save_budget_history: Whether to save budget consumption history
            budget_save_dir: Directory to save budget history
            verbose: Verbosity level
        """
        super(NSMDMPICallback, self).__init__(verbose)
        
        # Variation budgets
        self.V_R_init = V_R
        self.V_P_init = V_P
        self.V_pi_star_init = V_pi_star
        self.auto_estimate_budgets = auto_estimate_budgets
        self.budget_scale_factor = budget_scale_factor
        self.budget_tracker: Optional[VariationBudgetTracker] = None
        
        # Trust region config
        self.kappa_base = kappa_base
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.kappa_adaptive = kappa_adaptive
        self.trust_region_sensitivity = trust_region_sensitivity
        self.kappa_t = kappa_base
        
        # Regularization config (Temperature form η_t)
        self.lambda_base = lambda_base  # d_0 in paper
        self.lambda_min = lambda_min  # η_min
        self.lambda_max = lambda_max  # η_max
        self.lambda_adaptive = lambda_adaptive
        self.regularization_sensitivity = regularization_sensitivity  # d_1 in paper
        self.lambda_t = lambda_base
        
        # EMA tracking for drift (τ window)
        self.drift_ema = 0.0
        self.ema_tau = 0.1  # EMA coefficient (story.md: EMA_τ)
        
        # Drift estimation config
        self.drift_weights = drift_weights
        estimator_config = DriftEstimatorConfig(
            window_size=drift_window_size,
            min_samples=drift_min_samples,
            ema_alpha=0.01,
        )
        self.drift_estimator = CombinedDriftEstimator(estimator_config)
        
        # Entropy config
        self.adapt_entropy = adapt_entropy
        self._base_ent_coef = base_ent_coef
        self.min_ent_coef = min_ent_coef
        self.max_ent_coef = max_ent_coef
        self.current_ent_coef = base_ent_coef
        
        # Logging
        self.log_freq = log_freq
        self.save_budget_history = save_budget_history
        self.budget_save_dir = budget_save_dir
        
        # Runtime state
        self.algo_name: Optional[str] = None
        self.base_lr: float = 0.0
        self.current_lr: float = 0.0
        self.current_clip_range: float = 0.2
        self.current_target_kl: float = 0.01
        
        # Episode tracking for rollout-level updates
        self.rollout_count = 0
        self.last_rollout_drift = {'delta_R': 0.0, 'delta_P': 0.0, 'delta_C': 0.0}

    def _on_training_start(self) -> None:
        """Initialize budgets and base values from model."""
        # Detect algorithm
        self.algo_name = self.model.__class__.__name__.upper()
        
        # Get base learning rate (algorithm-specific)
        if self.algo_name == 'SAC':
            # SAC has separate optimizers for actor and critic
            self.base_lr = self.model.actor.optimizer.param_groups[0]["lr"]
        else:
            # PPO and other on-policy algorithms
            self.base_lr = self.model.policy.optimizer.param_groups[0]["lr"]
        self.current_lr = self.base_lr
        
        # Algorithm-specific initialization
        if self.algo_name == 'PPO':
            self._init_ppo()
        elif self.algo_name == 'SAC':
            self._init_sac()
        elif self.algo_name == 'TRPO':
            self._init_trpo()
        
        # Initialize variation budgets
        self._initialize_budgets()
        
        # Restore budget state from checkpoint if resuming
        if hasattr(self.model, '_resume_budget_state') and self.model._resume_budget_state:
            state = self.model._resume_budget_state
            if self.budget_tracker:
                self.budget_tracker.V_R_remaining = state.get('V_R_remaining', self.budget_tracker.V_R_remaining)
                self.budget_tracker.V_P_remaining = state.get('V_P_remaining', self.budget_tracker.V_P_remaining)
                self.budget_tracker.V_pi_star_remaining = state.get('V_pi_star_remaining', self.budget_tracker.V_pi_star_remaining)
            self.kappa_t = state.get('kappa_t', self.kappa_t)
            self.lambda_t = state.get('lambda_t', self.lambda_t)
            print(f"\n>>> [NS-MD-MPI] Restored budget state from checkpoint:")
            print(f"    V_R: {self.budget_tracker.V_R_remaining:.1f}, V_P: {self.budget_tracker.V_P_remaining:.1f}")
            print(f"    kappa_t: {self.kappa_t:.4f}, lambda_t: {self.lambda_t:.4f}")
            # Clear the attribute
            del self.model._resume_budget_state
        
        if self.verbose > 0:
            self._print_init_summary()
    
    def _init_ppo(self) -> None:
        """Initialize PPO-specific base values."""
        # Clip range
        if hasattr(self.model, 'clip_range'):
            clip_fn = self.model.clip_range
            if callable(clip_fn):
                self.kappa_base = clip_fn(1.0)
            else:
                self.kappa_base = float(clip_fn)
        self.current_clip_range = self.kappa_base
        self.kappa_t = self.kappa_base
        
        # Entropy coefficient
        if hasattr(self.model, 'ent_coef'):
            if self._base_ent_coef == 0.0:  # Auto-detect
                self._base_ent_coef = self.model.ent_coef
        self.current_ent_coef = self._base_ent_coef
    
    def _init_sac(self) -> None:
        """Initialize SAC-specific base values."""
        # SAC entropy coefficient (can be learned or fixed)
        if hasattr(self.model, 'ent_coef'):
            if self._base_ent_coef == 0.0:
                try:
                    self._base_ent_coef = float(self.model.ent_coef)
                except:
                    self._base_ent_coef = 0.01
        self.current_ent_coef = self._base_ent_coef
    
    def _init_trpo(self) -> None:
        """Initialize TRPO-specific base values."""
        # TRPO target KL
        if hasattr(self.model, 'target_kl'):
            self.kappa_base = self.model.target_kl
        self.current_target_kl = self.kappa_base
        self.kappa_t = self.kappa_base
    
    def _initialize_budgets(self) -> None:
        """Initialize variation budget tracker."""
        V_R = self.V_R_init
        V_P = self.V_P_init
        V_pi_star = self.V_pi_star_init
        
        # Auto-estimate from environment if requested
        if self.auto_estimate_budgets:
            try:
                env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
                if hasattr(env, 'get_drift_info'):
                    drift_info = env.get_drift_info()
                    # Extract drift config from first parameter
                    if 'parameters' in drift_info and len(drift_info['parameters']) > 0:
                        first_param = list(drift_info['parameters'].values())[0]
                        drift_config = first_param.get('config', {})
                        
                        # Estimate based on training horizon
                        total_timesteps = getattr(self.model, '_total_timesteps', 100000)
                        V_R, V_P, V_pi_star = estimate_budgets_from_drift_config(
                            drift_config,
                            total_timesteps,
                            scale_factor=self.budget_scale_factor,
                        )
                        
                        if self.verbose > 0:
                            print(f">>> [NS-MD-MPI] Auto-estimated budgets from drift config:")
                            print(f"    V_R={V_R:.2f}, V_P={V_P:.2f}, V_π*={V_pi_star:.2f}")
            except Exception as e:
                if self.verbose > 0:
                    print(f">>> [NS-MD-MPI] Could not auto-estimate budgets: {e}")
                    print(f"    Using manual values: V_R={V_R}, V_P={V_P}, V_π*={V_pi_star}")
        
        # Create tracker
        total_timesteps = getattr(self.model, '_total_timesteps', None)
        self.budget_tracker = VariationBudgetTracker(
            V_R=V_R,
            V_P=V_P,
            V_pi_star=V_pi_star,
            time_horizon=total_timesteps,
        )
    
    def _print_init_summary(self) -> None:
        """Print initialization summary."""
        print(f"\n{'='*60}")
        print(f"NS-MD-MPI Callback Initialized")
        print(f"{'='*60}")
        print(f"Algorithm: {self.algo_name}")
        print(f"\nVariation Budgets:")
        print(f"  V_R (Reward):     {self.budget_tracker.V_R_total:.2f}")
        print(f"  V_P (Transition): {self.budget_tracker.V_P_total:.2f}")
        print(f"  V_π* (Policy):    {self.budget_tracker.V_pi_star_total:.2f}")
        print(f"\nTrust Region:")
        print(f"  κ_base: {self.kappa_base:.4f}")
        print(f"  Adaptive: {self.kappa_adaptive}")
        print(f"  Sensitivity (α): {self.trust_region_sensitivity:.2f}")
        print(f"\nRegularization:")
        print(f"  λ_base: {self.lambda_base:.4f}")
        print(f"  Adaptive: {self.lambda_adaptive}")
        print(f"  Sensitivity (β): {self.regularization_sensitivity:.2f}")
        print(f"\nDrift Estimation:")
        print(f"  Weights (R,P,C): {self.drift_weights}")
        print(f"  Window size: {self.drift_estimator.config.window_size}")
        print(f"\nBase Hyperparameters:")
        print(f"  Learning Rate: {self.base_lr:.6f}")
        if self.algo_name == 'PPO':
            print(f"  Clip Range: {self.current_clip_range:.4f}")
            print(f"  Entropy Coef: {self.current_ent_coef:.4f}")
        elif self.algo_name == 'TRPO':
            print(f"  Target KL: {self.current_target_kl:.4f}")
        elif self.algo_name == 'SAC':
            print(f"  Entropy Coef: {self.current_ent_coef:.4f}")
        print(f"{'='*60}\n")
    
    def _on_step(self) -> bool:
        """Update drift estimators at each step."""
        # Get info from environment
        infos = self.locals.get('infos', [])
        if len(infos) > 0:
            info = infos[0]
            rewards = self.locals.get('rewards', [0])
            reward = rewards[0] if len(rewards) > 0 else 0
            
            # Update drift estimators (reward + transition)
            self.drift_estimator.update(reward, info)
            
            # Extract value predictions for CommutatorEstimator
            # PPO stores values in rollout buffer
            try:
                if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
                    buffer = self.model.rollout_buffer
                    if buffer.pos > 0:  # Has data
                        # Get latest value prediction and return
                        value_pred = buffer.values[buffer.pos - 1, 0] if buffer.pos > 0 else 0.0
                        returns = buffer.returns[buffer.pos - 1, 0] if hasattr(buffer, 'returns') and buffer.pos > 0 else value_pred
                        td_error = returns - value_pred
                        
                        # Update commutator estimator with TD error
                        self.drift_estimator.commutator_estimator.update(
                            value_pred=float(value_pred),
                            value_target=float(returns),
                            td_error=float(td_error)
                        )
            except Exception as e:
                # Silently skip if buffer not ready
                pass
        
        # Log periodically
        if self.n_calls % self.log_freq == 0:
            self._log_metrics()
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called at end of each rollout (episode collection).
        This is where we implement Algorithm 1 logic.
        """
        self.rollout_count += 1
        
        # 1. Estimate drift from current rollout
        delta_R = self.drift_estimator.reward_estimator.estimate_drift()
        delta_P = self.drift_estimator.transition_estimator.estimate_drift()
        delta_C = self.drift_estimator.commutator_estimator.estimate_commutator()
        
        # Store for logging
        self.last_rollout_drift = {
            'delta_R': delta_R,
            'delta_P': delta_P,
            'delta_C': delta_C,
        }
        
        # 2. Update variation budgets
        # Use commutator as proxy for policy variation (Option A)
        delta_pi = delta_C
        self.budget_tracker.update(
            delta_R=delta_R,
            delta_P=delta_P,
            delta_pi=delta_pi,
            timestep=self.num_timesteps,
        )
        
        # 3. Compute combined drift using Bellman Commutator formula (paper Eq. 559)
        # Δ̂_t = Δ̂_t^R + 2γB·Δ̂_t^P
        gamma = 0.99  # Discount factor
        B = 1.0  # Bound on Q-values (can be estimated from training)
        combined_drift = delta_R + 2 * gamma * B * delta_P
        
        # Update EMA of drift (paper: EMA_τ)
        self.drift_ema = (1 - self.ema_tau) * self.drift_ema + self.ema_tau * combined_drift
        
        # Get budget fraction for adaptive control
        budget_frac = self.budget_tracker.get_min_remaining_fraction()
        
        # 4. Compute adaptive trust region κ_t
        # FIXED: Now uses BOTH budget_fraction AND drift
        # Formula (matching docstring): κ_t = κ_0 * budget_frac * (1 + α * drift_ema)
        # - Shrinks as budgets deplete (budget_frac → 0)
        # - Expands temporarily when drift is high
        if self.kappa_adaptive:
            # Budget factor: shrink trust region as budgets deplete
            budget_factor = max(0.1, budget_frac)  # Floor at 10% to avoid complete collapse
            
            # Drift factor: allow larger steps when drift detected
            drift_factor = 1.0 + self.trust_region_sensitivity * self.drift_ema
            
            # Combined: trust region shrinks with budget but can expand with drift
            self.kappa_t = self.kappa_base * budget_factor * drift_factor
            self.kappa_t = np.clip(self.kappa_t, self.kappa_min, self.kappa_max)
        
        # 5. Compute adaptive regularization λ_t
        # FIXED: Now uses BOTH budget_fraction AND drift
        # More regularization when: (a) drift is high, OR (b) budgets depleted
        if self.lambda_adaptive:
            # Budget factor: increase regularization as budgets deplete
            budget_reg = 1.0 + (1.0 - budget_frac) * 0.5  # Up to 1.5x when depleted
            
            # Drift factor: increase regularization when drift is high
            drift_reg = 1.0 + self.regularization_sensitivity * self.drift_ema
            
            # Combined regularization
            self.lambda_t = self.lambda_base * budget_reg * drift_reg
            self.lambda_t = np.clip(self.lambda_t, self.lambda_min, self.lambda_max)
        
        # 6. Apply trust region and regularization via hyperparameters
        self._apply_trust_region()
        self._apply_regularization()
        
        # 7. Log budget exhaustion warning
        if self.budget_tracker.is_budget_exhausted(threshold=0.1):
            if self.verbose > 0:
                print(f"\n⚠️  [NS-MD-MPI] Warning: Variation budgets nearly exhausted!")
                print(f"    Remaining: {budget_frac:.1%}")
                print(f"    Trust region reduced to κ_t={self.kappa_t:.4f}\n")
    
    def _apply_trust_region(self) -> None:
        """Apply trust region constraint via algorithm-specific hyperparameters."""
        if self.algo_name == 'PPO':
            # PPO: Trust region via clip_range
            if hasattr(self.model, 'clip_range'):
                self.model.clip_range = lambda _progress: self.kappa_t
                self.current_clip_range = self.kappa_t
        
        elif self.algo_name == 'TRPO':
            # TRPO: Trust region via target_kl
            if hasattr(self.model, 'target_kl'):
                self.model.target_kl = self.kappa_t
                self.current_target_kl = self.kappa_t
    
    def _apply_regularization(self) -> None:
        """Apply regularization via entropy coefficient and learning rate."""
        # Adjust learning rate based on λ_t
        # Higher regularization → lower LR (more cautious updates)
        lr_multiplier = 1.0 / self.lambda_t
        new_lr = self.base_lr * lr_multiplier
        
        # Update learning rate for all optimizers (algorithm-specific)
        if self.algo_name == 'SAC':
            # SAC has separate optimizers for actor and critic
            for param_group in self.model.actor.optimizer.param_groups:
                param_group["lr"] = new_lr
            for param_group in self.model.critic.optimizer.param_groups:
                param_group["lr"] = new_lr
        else:
            # PPO and other on-policy algorithms
            for param_group in self.model.policy.optimizer.param_groups:
                param_group["lr"] = new_lr
        self.current_lr = new_lr
        
        # Adjust entropy coefficient (more exploration when drifting)
        if self.adapt_entropy and self._base_ent_coef > 0:
            # Higher drift → higher entropy (more exploration)
            combined_drift = (
                self.drift_weights[0] * self.last_rollout_drift['delta_R'] +
                self.drift_weights[1] * self.last_rollout_drift['delta_P'] +
                self.drift_weights[2] * self.last_rollout_drift['delta_C']
            )
            
            ent_multiplier = 1.0 + combined_drift
            new_ent = self._base_ent_coef * ent_multiplier
            new_ent = np.clip(new_ent, self.min_ent_coef, self.max_ent_coef)
            
            if hasattr(self.model, 'ent_coef'):
                self.model.ent_coef = new_ent
            self.current_ent_coef = new_ent
    
    def _log_metrics(self) -> None:
        """Log NS-MD-MPI metrics to logger."""
        # Budget metrics
        self.logger.record("nsmdmpi/V_R_remaining", self.budget_tracker.V_R_remaining)
        self.logger.record("nsmdmpi/V_P_remaining", self.budget_tracker.V_P_remaining)
        self.logger.record("nsmdmpi/V_pi_remaining", self.budget_tracker.V_pi_star_remaining)
        
        remaining_fracs = self.budget_tracker.get_remaining_fraction()
        self.logger.record("nsmdmpi/V_R_fraction", remaining_fracs[0])
        self.logger.record("nsmdmpi/V_P_fraction", remaining_fracs[1])
        self.logger.record("nsmdmpi/V_pi_fraction", remaining_fracs[2])
        self.logger.record("nsmdmpi/min_budget_fraction", self.budget_tracker.get_min_remaining_fraction())
        
        # Trust region
        self.logger.record("nsmdmpi/kappa_t", self.kappa_t)
        
        # Regularization
        self.logger.record("nsmdmpi/lambda_t", self.lambda_t)
        
        # Drift estimates
        self.logger.record("nsmdmpi/delta_R", self.last_rollout_drift['delta_R'])
        self.logger.record("nsmdmpi/delta_P", self.last_rollout_drift['delta_P'])
        self.logger.record("nsmdmpi/delta_C", self.last_rollout_drift['delta_C'])
        
        # Hyperparameters
        self.logger.record("nsmdmpi/learning_rate", self.current_lr)
        if self.algo_name == 'PPO':
            self.logger.record("nsmdmpi/clip_range", self.current_clip_range)
            self.logger.record("nsmdmpi/ent_coef", self.current_ent_coef)
        elif self.algo_name == 'TRPO':
            self.logger.record("nsmdmpi/target_kl", self.current_target_kl)
        elif self.algo_name == 'SAC':
            self.logger.record("nsmdmpi/ent_coef", self.current_ent_coef)
    
    def _on_training_end(self) -> None:
        """Save budget history at end of training."""
        if self.save_budget_history and self.budget_tracker is not None:
            import os
            os.makedirs(self.budget_save_dir, exist_ok=True)
            
            save_path = os.path.join(
                self.budget_save_dir,
                f"budget_history_{self.algo_name}_{self.num_timesteps}.json"
            )
            
            try:
                self.budget_tracker.save(save_path)
                if self.verbose > 0:
                    print(f"\n>>> [NS-MD-MPI] Budget history saved to {save_path}")
                    print(f"    Final budget utilization:")
                    summary = self.budget_tracker.get_summary()
                    print(f"      V_R: {summary['V_R_fraction_consumed']:.1%} consumed")
                    print(f"      V_P: {summary['V_P_fraction_consumed']:.1%} consumed")
                    print(f"      V_π*: {summary['V_pi_fraction_consumed']:.1%} consumed")
            except Exception as e:
                if self.verbose > 0:
                    print(f">>> [NS-MD-MPI] Could not save budget history: {e}")
    
    def __getstate__(self):
        """Exclude unpicklable objects when saving."""
        state = self.__dict__.copy()
        # Remove logger reference (it's not picklable)
        if 'logger' in state:
            del state['logger']
        return state
    
    def __setstate__(self, state):
        """Restore state when loading."""
        self.__dict__.update(state)
        # Logger will be re-initialized by parent class
