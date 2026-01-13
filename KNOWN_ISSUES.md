# Known Issues with New Environments

## ⚠️ Swimmer-v4 MuJoCo Compatibility Issue

**Problem:**
Swimmer-v4 has a MuJoCo XML schema incompatibility:
```
ValueError: XML Error: Schema violation: unrecognized attribute: 'collision'
```

**Root Cause:**
- Older gymnasium MuJoCo envs (v4) may have XML that's incompatible with newer MuJoCo versions
- Swimmer v5 doesn't exist (only v2, v3, v4 available)

**Workaround Options:**

### Option 1: Skip Swimmer (Recommended for now)
```bash
# Use these envs for tuning instead:
bash scripts/tune_moderate_hopper.sh
bash scripts/tune_moderate_halfcheetah.sh
bash scripts/tune_moderate_walker2d.sh    # Should work
bash scripts/tune_moderate_humanoid.sh    # Should work
```

### Option 2: Downgrade MuJoCo (Not recommended)
```bash
pip install mujoco==2.3.0  # Older version
```
May cause other compatibility issues.

### Option 3: Use Swimmer-v3
Edit configs to use `Swimmer-v3` instead. May have different dynamics.

## ✅ Working Environments

These should work without issues:
- **Hopper-v4** ✅
- **HalfCheetah-v4** ✅  
- **Walker2d-v4** ✅ (likely)
- **Humanoid-v4** ✅ (likely)
- **LunarLander-v2** ✅

## 🔧 If Walker2D or Humanoid Also Fail

If you get similar XML errors, use v3:
```bash
# Update configs
find configs -name "*walker2d*.yaml" -exec sed -i 's/Walker2d-v4/Walker2d-v3/g' {} \;
find configs -name "*humanoid*.yaml" -exec sed -i 's/Humanoid-v4/Humanoid-v3/g' {} \;
```

## 📝 Recommendation

**For now, focus tuning on:**
1. Hopper (known to work)
2. HalfCheetah (known to work)
3. LunarLander (known to work)
4. Walker2D (test first)

**Skip Swimmer** until MuJoCo/Gymnasium compatibility is resolved.
