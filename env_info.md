# Environment Information Reference

Comprehensive specifications for all supported non-stationary RL environments.

---

## 1. CartPole-v1

### Task Description
Balance a pole on a moving cart by applying left/right forces. The pole starts upright and the goal is to prevent it from falling over.

### State Space
**Type**: Box (Continuous)  
**Dimension**: 4  
**Components**:
- Cart Position: [-4.8, 4.8]
- Cart Velocity: [-∞, ∞]
- Pole Angle: [-0.418 rad, 0.418 rad] (~±24°)
- Pole Angular Velocity: [-∞, ∞]

### Action Space
**Type**: Discrete(2)
- 0: Push cart to the left
- 1: Push cart to the right

### Reward Structure
- **Reward per step**: +1 (while pole is upright)
- **Maximum reward**: **500** (hard cap at 500 timesteps)
- **Episode termination**: Pole angle > ±12° or cart position > ±2.4

### Driftable Parameters (Our Wrapper)
- `gravity`: 9.8 m/s² (default)
- `masscart`: 1.0 kg (cart mass)
- `masspole`: 0.1 kg (pole mass)
- `length`: 0.5 m (pole half-length)

---

## 2. MountainCar-v0

### Task Description
Drive an underpowered car up a steep mountain. The car must rock back and forth to build momentum to reach the goal at the top.

### State Space
**Type**: Box (Continuous)  
**Dimension**: 2  
**Components**:
- Position: [-1.2, 0.6] (goal at 0.5)
- Velocity: [-0.07, 0.07]

### Action Space
**Type**: Discrete(3)
- 0: Accelerate to the left
- 1: Don't accelerate
- 2: Accelerate to the right

### Reward Structure
- **Reward per step**: -1 (penalty for each timestep)
- **Goal reward**: Reach position >= 0.5
- **Maximum episode length**: 200 steps
- **Best possible reward**: ~-110 (reaching goal quickly)

### Driftable Parameters (Our Wrapper)
- `gravity`: 0.0025 (default)
- `force`: 0.001 (car engine power)
- `goal_position`: 0.5 (goal location)

---

## 3. Hopper-v4 (MuJoCo)

### Task Description
Make a 2D one-legged robot (hopper) move forward by hopping. The robot must maintain balance while propelling itself forward.

### State Space
**Type**: Box (Continuous)  
**Dimension**: 11  
**Components**:
- z-coordinate (height): [0.7, ∞]
- Joint angles: 3 values (thigh, leg, foot)
- Velocities: 6 values (body + joints)
- **Note**: Excludes x-position (to avoid learning "run fast then fall")

### Action Space
**Type**: Box (Continuous)  
**Dimension**: 3  
**Range**: [-1, 1] for each actuator
- Thigh joint torque
- Leg joint torque  
- Foot joint torque

### Reward Structure
**Formula**: `healthy_reward + forward_reward - ctrl_cost`
- **healthy_reward**: +1.0 per timestep (if alive)
- **forward_reward**: velocity in x-direction
- **ctrl_cost**: 0.001 × ||action||²
- **Reward threshold (solved)**: **3800.0**
- **Maximum episode length**: 1000 steps
- **Termination**: Height < 0.7 or |angle| > 0.2 rad

### Driftable Parameters (Our Wrapper)
- `friction`: 0.9 (floor friction coefficient)
- `damping`: 1.0 (joint damping multiplier)
- `mass_scale`: 1.0 (body mass multiplier)
- `torso_length`: 1.0 (torso size multiplier)

---

## 4. HalfCheetah-v4 (MuJoCo)

### Task Description
Make a 2D cheetah robot run forward as fast as possible. Unlike Hopper, HalfCheetah cannot fall over (no termination condition).

### State Space
**Type**: Box (Continuous)  
**Dimension**: 17  
**Components**:
- Root element velocities: 2 values
- Joint angles: 6 values (back thigh, shin, front thigh, shin, foot, etc.)
- Joint velocities: 6 values
- **Note**: Excludes x-position

### Action Space
**Type**: Box (Continuous)  
**Dimension**: 6  
**Range**: [-1, 1] for each actuator
- 6 joint torques (spine, back/front thighs, shins, feet)

### Reward Structure
**Formula**: `forward_reward - ctrl_cost`
- **forward_reward**: velocity in x-direction
- **ctrl_cost**: 0.1 × ||action||²
- **Reward threshold (solved)**: **4800.0**
- **Maximum episode length**: 1000 steps
- **No termination condition** (always runs 1000 steps)

### Driftable Parameters (Our Wrapper)
- `friction`: 0.4 (floor friction coefficient)
- `damping`: 1.0 (joint damping multiplier)
- `mass_scale`: 1.0 (body mass multiplier)
- `gravity`: -9.81 m/s² (gravitational acceleration)

---

## 5. FrozenLake-v1

### Task Description
Navigate from start (S) to goal (G) on a 4×4 frozen lake grid without falling into holes (H). The ice is slippery, so actions are stochastic.

### State Space
**Type**: Discrete(16)
- Single integer representing current grid position (0-15)
- Grid layout (4×4):
```
S F F F
F H F H  
F F F H
H F F G
```
- S: Start, F: Frozen (safe), H: Hole, G: Goal

### Action Space
**Type**: Discrete(4)
- 0: Left
- 1: Down
- 2: Right
- 3: Up
- **Note**: With default `is_slippery=True`, action succeeds only 1/3 of the time

### Reward Structure
- **Reward**: 0 for all steps except:
  - +1 for reaching goal (G)
  - 0 for falling in hole (H) - episode ends
- **Maximum reward**: **1** (binary: success or failure)
- **Maximum episode length**: 100 steps

### Driftable Parameters (Our Wrapper)
- `slip_prob`: Probability of slipping (default ~0.66)
- `reward_scale`: Multiplier for goal reward

---

## 6. MiniGrid-Dynamic-Obstacles-8x8-v0

### Task Description
Navigate a partially observable 8×8 grid world to reach a green goal square while avoiding moving obstacles. Agent only sees a 7×7 region around itself.

### State Space
**Type**: Dict
- **image**: Box(7, 7, 3) - Partial observation around agent
  - Encodes: object type, color, state
- **direction**: Discrete(4) - Agent's facing direction
- **mission**: Text string describing the goal

### Action Space
**Type**: Discrete(7)
- 0: Turn left
- 1: Turn right
- 2: Move forward
- 3: Pick up object
- 4: Drop object
- 5: Toggle/activate object
- 6: Done (declare task complete)

### Reward Structure
- **Sparse reward**: 0 for all steps
- **Goal reward**: 1 - 0.9 × (step_count / max_steps)
  - Encourages reaching goal quickly
- **Maximum reward**: **1.0**
- **Default max steps**: 64 (8×8 grid)

### Driftable Parameters (Our Wrapper)
- `reward_scale`: Multiplier for rewards
- `max_steps`: Episode length
- `num_obstacles`: Number of dynamic obstacles (default: 4)

---

## 7. Procgen (Coin Run)

### Task Description
Procedurally generated platformer game where the agent collects coins while avoiding or jumping over enemies. Each level is randomly generated.

### State Space
**Type**: Box(64, 64, 3) - RGB Image  
**Dimension**: (64, 64, 3)
- Pixels representing the game screen
- **Procedurally generated**: Each episode has a different level layout

### Action Space
**Type**: Discrete(15)
Standard game controller actions:
- Directional: Left, Right, Up, Down
- Buttons: A, B (jump, etc.)
- Combinations: Left+A, Right+A, etc.
- **Effective actions in CoinRun**: ~5-7 meaningful actions

### Reward Structure
- **Coins collected**: +1 per coin
- **Level completion**: +10 (bonus)
- **No fixed maximum**: Depends on procedurally generated level
- **Typical range**: 0-15 per episode
- **Average episode length**: ~500-1000 steps

### Driftable Parameters
⚠️ **Limited drift applicability**:
- Procgen environments are **inherently non-stationary** (procedural generation)
- No physics parameters to modify
- Possible drift: difficulty level, reward scaling

---

## 8. LunarLander-v3

### Task Description
Land a lunar module safely on a landing pad between two flags. Control main engine and side thrusters to navigate and land softly.

### State Space
**Type**: Box (Continuous)  
**Dimension**: 8  
**Components**:
- Position: (x, y)
- Velocity: (vx, vy)
- Angle: θ
- Angular velocity: ω
- Leg contact: (left_leg, right_leg) - binary

### Action Space
**Type**: Discrete(4)
- 0: Do nothing
- 1: Fire left orientation engine
- 2: Fire main engine
- 3: Fire right orientation engine

### Reward Structure
**Complex shaped reward**:
- Moving toward/away from pad: -/+ 
- Crash: -100
- Safe landing: +100
- Leg contact: +10 each
- Engine usage: -0.3 per frame for main, -0.03 for side
- **Solved threshold**: **200**
- **Maximum possible**: ~280 (perfect landing)

### Driftable Parameters (Our Wrapper)
- `gravity`: -10.0 (default, Earth-like)
- `wind_power`: 0.0 to 20.0 (lateral wind force)
- `turbulence_power`: 0.0 to 2.0 (observation noise)

---

## Summary Table

| Environment | State Dim | Action Type | Max Reward | Difficulty |
|-------------|-----------|-------------|------------|------------|
| **CartPole-v1** | 4 | Discrete(2) | 500 | ⭐ Easy |
| **MountainCar-v0** | 2 | Discrete(3) | ~-110 | ⭐⭐ Medium |
| **Hopper-v4** | 11 | Box(3) | ~3800+ | ⭐⭐⭐⭐ Hard |
| **HalfCheetah-v4** | 17 | Box(6) | ~4800+ | ⭐⭐⭐⭐ Hard |
| **FrozenLake-v1** | Discrete(16) | Discrete(4) | 1 | ⭐⭐ Medium |
| **MiniGrid-8x8** | Dict (7×7×3) | Discrete(7) | 1 | ⭐⭐⭐ Medium-Hard |
| **Procgen** | (64,64,3) | Discrete(15) | Variable | ⭐⭐⭐⭐ Hard |
| **LunarLander-v3** | 8 | Discrete(4) | 200-280 | ⭐⭐⭐ Medium-Hard |

---

## Notes

### Reward Types
- **Fixed Maximum**: CartPole (500), FrozenLake (1), MiniGrid (1)
- **Threshold-based**: Hopper (3800), HalfCheetah (4800), LunarLander (200)
- **Unbounded**: MuJoCo environments can theoretically achieve higher rewards
- **Variable**: Procgen (depends on procedural generation)

### State Space Complexity
- **Simple**: CartPole (4D), MountainCar (2D)
- **Medium**: LunarLander (8D), Hopper (11D)
- **Complex**: HalfCheetah (17D), MiniGrid (partial observation)
- **Very Complex**: Procgen (high-dimensional image observation)

### Control Difficulty
1. **Discrete, Simple**: CartPole, FrozenLake
2. **Discrete, Complex**: LunarLander, MiniGrid
3. **Continuous, High-Dimensional**: MuJoCo (Hopper, HalfCheetah)
4. **Vision-based**: Procgen

---

## Implementation Status

| Environment | Wrapper | Configs | Tested |
|-------------|---------|---------|--------|
| CartPole | ✅ | ✅ (11+ configs) | ✅ |
| MountainCar | ✅ | ❌ | ❌ |
| Hopper | ✅ | ✅ (3 configs) | 🟡 In Progress |
| HalfCheetah | ✅ | ❌ | ❌ |
| FrozenLake | ✅ | ❌ | ❌ |
| MiniGrid | ✅ | ✅ (1 config) | ❌ |
| Procgen | 🟡 Partial | ❌ | ❌ |
| LunarLander | ✅ | ✅ (3 configs) | ❌ |

**Legend**: ✅ Complete | 🟡 Partial | ❌ Not Yet

---

*Last updated: 2026-01-10*
