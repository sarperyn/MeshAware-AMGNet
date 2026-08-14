# ANN experiment commands

## Prepare the epsilon dataset

```bash
export AMG_PYTHON=/Users/sarperyurtseven/Desktop/sarper/.envs/pytorch/bin/python

"$AMG_PYTHON" experiments/epsilon_policy/prepare_data.py \
  --config experiments/epsilon_policy/configs/medium_default.json
```

Creates the cleaned solve-time-only dataset and interpolation splits in
`results/epsilon_policy/prepared/`.

## Generate the epsilon landscape

```bash
export AMG_PYTHON=/Users/sarperyurtseven/Desktop/sarper/.envs/pytorch/bin/python

"$AMG_PYTHON" experiments/epsilon_policy/epsilon_landscape.py \
  --config experiments/epsilon_policy/configs/medium_default.json
```

Writes solve-time landscape tables and plots to
`results/epsilon_policy/landscape/`.

## Epsilon interpolation: separate model per mesh family

```bash
export AMG_PYTHON=/Users/sarperyurtseven/Desktop/sarper/.envs/pytorch/bin/python

"$AMG_PYTHON" experiments/epsilon_policy/train_interpolation.py \
  --config experiments/epsilon_policy/configs/interpolation.json \
  --mode separate
```

Trains three independent models and writes results to
`results/epsilon_policy/interpolation/separate/`.

## Epsilon interpolation: one joint model

```bash
export AMG_PYTHON=/Users/sarperyurtseven/Desktop/sarper/.envs/pytorch/bin/python

"$AMG_PYTHON" experiments/epsilon_policy/train_interpolation.py \
  --config experiments/epsilon_policy/configs/interpolation.json \
  --mode joint
```

Trains one model across all three default-Boomer mesh families and writes to
`results/epsilon_policy/interpolation/joint/`.

Append `--resume` to either interpolation command after an interrupted run.

## Explicit epsilon and matrix-awareness ablation

```bash
export AMG_PYTHON=/Users/sarperyurtseven/Desktop/sarper/.envs/pytorch/bin/python

"$AMG_PYTHON" experiments/epsilon_policy/train_ablation.py \
  --config experiments/epsilon_policy/configs/ablation.json
```

Runs the four joint-model variants and writes the 2×2 comparison to
`results/epsilon_policy/ablation/`. The no-matrix controls use a fixed zero
view with the same CNN capacity. Append `--resume` after interruption.
