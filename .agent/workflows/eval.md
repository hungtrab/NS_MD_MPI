---
description: Đánh giá model đã train và tính dynamic regret
---

# Evaluation Workflow

## Evaluation cơ bản

// turbo
1. Đánh giá model:
```bash
python scripts/eval.py --model models/<MODEL_NAME>.zip --config <CONFIG_PATH>
```

## Evaluation với Dynamic Regret

// turbo
2. Đánh giá với dynamic regret calculation:
```bash
python scripts/eval.py --model models/<MODEL_NAME>.zip --regret --eval-interval 5000
```

## Quay video

// turbo
3. Quay video demo:
```bash
python scripts/render.py --model models/<MODEL_NAME>.zip --episodes 1
```

## Output

- Kết quả được log ra console và WandB
- Videos được lưu tại `videos/`
- Models được lưu tại `models/`
