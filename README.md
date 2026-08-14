# AMG-ThetaNet

AMG-ThetaNet selects the HYPRE BoomerAMG strong-connection threshold $\theta$
for heterogeneous diffusion problems, and measures how that choice interacts
with the finite-element mesh. C++ drivers built on deal.II assemble the model
problem $-\nabla\cdot(\mu(x,y)\nabla u) = f$ on $\Omega = (-1,1)^2$ over
quadrilateral, conforming simplex, simplex discontinuous-Galerkin, and
polygonal discretizations, solve the resulting systems with preconditioned
conjugate gradients, and record per-trial convergence and timing data together
with the sparse operator.

A Python pipeline turns those artifacts into a supervised learning problem: a
convolutional regression model consumes a fixed-size pooled view of the sparse
matrix conditioned on $(h, \theta)$ and predicts the residual-based convergence
factor $\rho$. Minimizing the predicted $\rho$ over a candidate grid yields a
recommended HYPRE BoomerAMG strong threshold, which can then be fed straight
back into the solver. The experimental design follows
[*Accelerating Algebraic Multigrid Methods via Artificial Neural
Networks*](https://arxiv.org/abs/2111.01629).

## Contents

- [Capabilities](#capabilities)
- [Dependencies](#dependencies)
- [Clone and build](#clone-and-build)
- [Numerical experiment: medium sweep](#numerical-experiment-medium-sweep)
- [ANN workflow](#ann-workflow)
- [Configuration guide](#configuration-guide)
- [Repository structure](#repository-structure)
- [Scientific outputs and reproducibility](#scientific-outputs-and-reproducibility)
- [Citation and license](#citation-and-license)

## Capabilities

Finite-element assembly and solves (C++):

| Driver | Mesh families | Discretization | AMG backends |
| --- | --- | --- | --- |
| `meshaware_diffusion_dealii` | `quadrilateral`, `simplex` | $Q_1$ / $P_1$ continuous Lagrange | BoomerAMG (default profile) |
| `meshaware_diffusion_simplex_dg` | `simplex-dg` | $P_1$ symmetric interior-penalty DG | BoomerAMG (default profile) |
| `meshaware_diffusion_polydeal`| `polygonal` | degree-1 modal DG on agglomerated polygons (`FE_AggloDGP`) | BoomerAMG (`default` or `polygonal-nodal` profile), or explicit PolyDeal agglomeration multigrid |

Common to all drivers:

- Four coefficient patterns (`vertical_split`, `checkerboard_2x2`,
  `vertical_stripes_4`, `checkerboard_4x4`) with contrast $10^{\varepsilon}$
  applied to the `white` or `gray` region.
- Preconditioned CG with configurable relative/absolute tolerances and
  iteration cap; four relaxation choices (`chebyshev`, `damped-jacobi`,
  `l1-symmetric-gauss-seidel`, `symmetric-gauss-seidel`).
- Batched execution over a $\theta$ grid for one assembled operator
  (`--theta-values`, `--repeats`, `--warmup-runs`), plus assembly-only export
  (`--assemble-only`).
- JSON trial records with separated assembly / AMG-setup / solve timings, CG
  iteration counts, AMG level and complexity data, discretization error norms,
  and the residual-based convergence factor.
- PETSc binary matrix export, converted to compressed SciPy CSR `.npz` by the
  experiment runner.

Only the polygonal driver exposes `--amg-backend` and `--boomeramg-profile`.
The `polygonal-nodal` BoomerAMG profile enables HYPRE nodal coarsening with a
near-nullspace and is restricted to polygonal-only experiments; the
`polydeal-agglomeration` backend builds an explicit nested polygon
agglomeration hierarchy instead of BoomerAMG and ignores $\theta$ (the value is
still recorded). The polygonal driver's `--oracle` mode additionally solves the
native deal.II system directly and compares matrices, right-hand sides, and
solutions; it is limited to `--level <= 6`.

Configuration-driven sweeps and ML (Python):

- Expansion of a versioned JSON experiment grid into trials grouped by matrix
  identity, one solver invocation per matrix, with resume support.
- Per-family CSV reports and optimal-$\theta$ summaries.
- Leakage-safe ML index construction: frozen dataset snapshots, pooled
  fixed-size matrix views, and grouped stratified train/validation/test splits
  keyed by matrix content checksum.
- CNN training for convergence-factor regression, with deterministic seeding,
  early stopping, and exact resume from the last checkpoint.
- Locked held-out evaluation reporting regression metrics, grouped bootstrap
  confidence intervals, and $\theta$-selection decision quality.
- Single-matrix inference producing a recommended $\theta$, and an end-to-end
  workflow that assembles, predicts, solves, and compares predicted against
  measured $\rho$.

## Dependencies

### Core build tools

| Requirement | Mandatory | Why | Reference |
| --- | --- | --- | --- |
| C++17 compiler | Yes | `meshaware_core` requests `cxx_std_17` | — |
| CMake ≥ 3.20 | Yes | Enforced by `cmake_minimum_required` in [CMakeLists.txt](CMakeLists.txt) | [cmake.org](https://cmake.org/download/) |
| MPI | Yes | All three drivers construct `Utilities::MPI::MPI_InitFinalize` at start-up, and deal.II/PETSc are linked through it. Runs are single-rank. | [Open MPI](https://www.open-mpi.org/) / [MPICH](https://www.mpich.org/) |

### Required C++ scientific libraries

| Requirement | Mandatory | Why | Reference |
| --- | --- | --- | --- |
| deal.II ≥ 9.7 | Yes, for every driver | `find_package(deal.II 9.7)`; provides meshes, elements, quadrature, and the PETSc wrappers. If it is not found, all C++ targets are skipped and only the Python tooling is usable. | [dealii.org](https://www.dealii.org/) · [install guide](https://www.dealii.org/current/readme.html) |
| PETSc | Yes | deal.II **must** be configured with PETSc support — CMake raises `FATAL_ERROR` when `DEAL_II_WITH_PETSC` is off. All linear algebra in [src/common/petsc_amg.cpp](src/common/petsc_amg.cpp) uses `KSP`/`PC`. | [petsc.org install](https://petsc.org/release/install/) |
| HYPRE (BoomerAMG) | Yes | The solver sets `PCHYPRE` + `PCHYPRESetType("boomeramg")`, so the PETSc build must include HYPRE (`--download-hypre` or `--with-hypre`). | [HYPRE docs](https://hypre.readthedocs.io/en/latest/) |

### polygonal / PolyDeal dependencies

Required only when building `meshaware_diffusion_polydeal`
(`MESHAWARE_BUILD_POLYDEAL=ON`, the default, plus a valid `POLYDEAL_ROOT`).

| Requirement | Why | Reference |
| --- | --- | --- |
| PolyDeal | Supplies `agglomeration_handler.h` and the `polydeal` library. CMake searches `${POLYDEAL_ROOT}/include` and `${POLYDEAL_ROOT}/build-unified/source` or `${POLYDEAL_ROOT}/build/source`. If either is missing, the polygonal target is skipped with a status message and the rest of the build proceeds. | [github.com/fdrmrc/Polydeal](https://github.com/fdrmrc/Polydeal) |
| Trilinos | The PolyDeal agglomeration multigrid backend requires deal.II built with Trilinos; CMake raises `FATAL_ERROR` when `DEAL_II_WITH_TRILINOS` is off. | [trilinos.github.io](https://trilinos.github.io/) |
| SuiteSparse / UMFPACK | Used by the polygonal driver's `--oracle` validation path, which calls `SparseDirectUMFPACK` to produce a reference direct solution. Not needed for ordinary solves. | [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) |

### Python dependencies


Python **3.10 or newer** is required


Example setup (adjust the PyTorch install command for your platform and
accelerator using the official selector):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn torch matplotlib
```

All Python commands below assume this environment is active and are run from
the repository root.

## Clone and build

```bash
git clone https://github.com/sarperyn/AMG-ThetaNet.git
cd AMG-ThetaNet
```

Configure. `DEAL_II_DIR` must point at the directory containing
`deal.IIConfig.cmake` (usually `<prefix>/lib/cmake/deal.II`), and
`POLYDEAL_ROOT` at the PolyDeal source tree you have already built:

```bash
cmake -S . -B build-unified \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DDEAL_II_DIR=/path/to/dealii/lib/cmake/deal.II \
  -DPOLYDEAL_ROOT=/path/to/Polydeal
```

Build:

```bash
cmake --build build-unified -j
```

CMake options:

| Option | Default | Effect |
| --- | --- | --- |
| `MESHAWARE_BUILD_DEALII` | `ON` | Build the deal.II drivers. With `OFF`, no C++ target is produced. |
| `MESHAWARE_BUILD_POLYDEAL` | `ON` | Attempt to build the polygonal driver. Has no effect unless deal.II is found and PolyDeal is located under `POLYDEAL_ROOT`. Set `OFF` to skip it explicitly. |
| `BUILD_TESTING` | `ON` (set by `include(CTest)`) | Registers the CTest suite. **In the current tree this must be `OFF`**: `CMakeLists.txt` line 16 references `tests/cpp/coefficient_patterns_test.cpp`, which is not present, so configuration fails at the generate step with `Cannot find source file`. |

Targets produced by a full build:

- `meshaware_core` — header-only interface library ([include/meshaware/](include/meshaware/)).
- `meshaware_petsc` — static library with the PETSc/BoomerAMG solver and record writer.
- `meshaware_diffusion_dealii` — quadrilateral and conforming simplex driver.
- `meshaware_diffusion_simplex_dg` — simplex SIPG driver.
- `meshaware_diffusion_polydeal` — polygonal driver (only when PolyDeal is found).

Executables land directly in the build directory, e.g.
`build-unified/meshaware_diffusion_dealii`. Each accepts `--help`:

```bash
./build-unified/meshaware_diffusion_dealii --help
```

`build-unified` is used consistently below because it is the default
`--build-dir` of [scripts/run_predicted_amg.py](scripts/run_predicted_amg.py).
[scripts/run_experiments.py](scripts/run_experiments.py) instead defaults to
`build`, so `--build-dir build-unified` is passed explicitly to it.

## Numerical experiment: medium sweep

[configs/medium.json](configs/medium.json) is layered on top of
[configs/common.json](configs/common.json): the loader reads `common.json` from
the same directory and lets the experiment file override individual keys. The
resulting grid is:

| Setting | Value | Source |
| --- | --- | --- |
| Mesh families | `quadrilateral`, `simplex`, `polygonal` | `medium.json` |
| Refinement levels | 3, 5, 8, 10 (nominal $h = 2^{-\text{level}}$) | `medium.json` |
| $\theta$ | 10 values, evenly spaced from 0.02 to 0.90 | `medium.json` |
| Coefficient patterns | `vertical_split`, `checkerboard_2x2`, `vertical_stripes_4`, `checkerboard_4x4` | `common.json` |
| $\varepsilon$ (contrast $10^{\varepsilon}$) | 0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.5, 5.0, 7.0, 9.5 | `common.json` |
| High-coefficient region | `white` | `common.json` |
| Repeats | 1 per $(\text{matrix}, \theta)$, plus 1 discarded warm-up | `medium.json` / `common.json` |
| AMG | BoomerAMG, `default` profile, `symmetric-gauss-seidel` relaxation, Jacobi damping 2/3 | `common.json` |
| CG tolerances | rtol $10^{-8}$, atol $10^{-50}$, at most 10000 iterations | `common.json` |
| Matrix output | `scipy_csr_npz`, saved | `common.json` |

`--dry-run` reports the expansion without running anything:

```bash
python scripts/run_experiments.py \
  --config configs/medium.json \
  --build-dir build-unified \
  --output-root datasets \
  --dry-run
```

For this configuration that prints **5760 expanded trials over 576 distinct
matrices**. Running the sweep in full is expensive: it assembles and solves at
level 10 for three mesh families, and every matrix is written to disk.

A single trial, useful as a smoke check. `--limit` also switches the runner
from batched execution to one solver invocation per trial
(`execution_mode: single_trial_debug`):

```bash
python scripts/run_experiments.py \
  --config configs/medium.json \
  --build-dir build-unified \
  --output-root datasets \
  --limit 1
```

One mesh family only. `--mesh-family` may be repeated; valid choices are
`quadrilateral`, `simplex`, `simplex-dg`, and `polygonal`, intersected with the
families the configuration declares:

```bash
python scripts/run_experiments.py \
  --config configs/medium.json \
  --build-dir build-unified \
  --output-root datasets \
  --mesh-family simplex
```

The complete sweep (expensive — hours to days depending on hardware):

```bash
python scripts/run_experiments.py \
  --config configs/medium.json \
  --build-dir build-unified \
  --output-root datasets
```

Resume semantics: by default a matrix group whose trial records all exist is
skipped, and partially complete groups are resumed by passing
`--skip-existing-records` to the driver. Matrices that are already stored are
not rewritten. `--overwrite-records` reruns everything and regenerates the
operators.

Outputs are written under `<output-root>/<config name>`, i.e.
`datasets/medium` for this configuration:

```
datasets/medium/
├── manifest.json                     # source config, fully expanded config, trial/matrix counts
├── quadrilateral/
│   ├── records/<sample_id>.json      # one trial record per (matrix, theta, repeat)
│   ├── matrices/<matrix_id>.npz      # compressed CSR operator, one per matrix
│   ├── diffusion_reports/trials.csv  # flattened trial table
│   └── summaries/optimal_theta.csv   # best theta per matrix
├── simplex/…
└── polygonal/…
```

`matrix_id` encodes the mesh family, level, pattern, $\varepsilon$, and
high region (for example `simplex_l3_vertical_split_e0_high_white`);
`sample_id` appends the $\theta$ value and repeat index. Per-family
subdirectories are the default and can be turned off with
`"family_subdirectories": false` for single-family configurations, as in
[configs/medium_simplex_dg.json](configs/medium_simplex_dg.json).

## ANN workflow

The model is a convolutional regressor
([python/meshaware_ml/model.py](python/meshaware_ml/model.py)). Three
convolution layers of width 40 followed by max-pooling and dropout compress a
$3 \times 100 \times 100$ pooled view of the sparse operator into a 128-dimensional
embedding; that embedding is concatenated with two scalars, $-\log_2 h$ and
$\theta$, passed through five dense layers of width 128, and mapped to a single
output. The training target is `rho_mean`, the residual-based convergence
factor measured by the solver. One matrix view therefore serves every $\theta$
sample for that matrix, and $\theta$ is selected at inference by evaluating the
model over a candidate grid and taking the argmin of the predicted $\rho$.

The pipeline accepts only `simplex` and `polygonal` data, enforced in the
dataset validator, the snapshot capture, the inference guards, and both CLIs.
This is the scope of the frozen `paper_v1` dataset rather than a missing
feature:

- **Quadrilateral is held out deliberately.** The medium sweep does generate
  quadrilateral operators; they are kept out of training so they can serve as
  the unseen-discretization probe — train on simplex and polygonal, test on
  quadrilateral — which asks whether the model generalizes across
  discretizations or merely recognizes families it has already seen.
- **`simplex-dg` postdates the frozen dataset.** Its records land in a separate
  tier (`datasets/medium-simplex-dg`, flat layout), which `--tier
  {small,medium}` does not reach.
- **Nodal-BoomerAMG and PolyDeal-agglomeration records are excluded** because
  the preconditioner changes the measured $\rho$ for an identical
  $(\text{matrix}, \theta)$ pair. Combining them would require an explicit
  backend input and a new dataset and model version.

### A. Generate numerical data

The pipeline consumes the trial records and finalized `.npz` operators produced
by [the numerical experiment step](#numerical-experiment-medium-sweep). It
looks for them at `<dataset-root>/<tier>/<mesh-family>/{records,matrices}`,
where `tier` is `small` or `medium` — that is, the `name` field of
[configs/small.json](configs/small.json) and
[configs/medium.json](configs/medium.json). Run at least one of those sweeps
for the simplex and polygonal families before continuing.

### B. Build the ML index

[scripts/build_ml_pipeline.py](scripts/build_ml_pipeline.py) freezes the
dataset, builds matrix views, and produces the canonical leakage-safe index:

```bash
python scripts/build_ml_pipeline.py \
  --dataset-root datasets \
  --output-root datasets/ml \
  --tier small --tier medium \
  --mesh-family simplex --mesh-family polygonal
```

`--tier` and `--mesh-family` are repeatable and default to both values each.
`--snapshot PATH` reuses an existing frozen snapshot instead of capturing a new
one, and `--report PATH` relocates the generated Markdown report.

What it does:

1. **Snapshot.** Records path, size, and mtime of every matrix and record file
   before reading any content, then verifies those stats again while reading.
   Staging `.petsc` files are listed but excluded.
2. **Views.** Each operator is pooled to the versioned `paper_v1` spec: a
   $100 \times 100$ grid of balanced index blocks with three channels
   (`positive_max`, `negative_max`, `sum`), `count_average` reduction,
   `signed_log1p_maxabs` normalization, `float32`. Views are keyed by the
   SHA-256 of the source matrix, so identical operators are pooled once and
   reused.
3. **Index and splits.** Samples are grouped by that matrix checksum and
   assigned to `train` / `validation` / `test` at 0.85 / 0.05 / 0.10 with seed
   2026, balancing the marginals of mesh family, level, family × level,
   pattern, $\varepsilon$, and $\theta$. Because assignment is per
   checksum, no matrix can appear in two partitions. Existing assignments are
   preserved across reruns unless `--reset-splits` is passed.

Artifacts, relative to `--output-root` (default `datasets/ml`):

| Path | Contents |
| --- | --- |
| `snapshots/<snapshot_id>.json` | Frozen file inventory and digest |
| `features/paper_v1/<sha256>.npz` | One pooled view per unique matrix checksum |
| `manifests/features-<snapshot_id>.json` | Feature cache manifest and matrix references |
| `index/paper_v1/samples.jsonl` | One row per $(\text{matrix}, \theta)$ sample with its split |
| `index/paper_v1/splits.json` | Checksum → split assignments, seed, ratios, statistics |
| `index/paper_v1/summary.json` | Audit counts, split statistics, target-ambiguity summary |
| `audits/phase_1_2.json` | Machine-readable audit |

The Markdown report defaults to `reports/ml_phase_1_2_report.md`.

### C. Train the CNN

[scripts/train_rho_cnn.py](scripts/train_rho_cnn.py) trains on the `train` and
`validation` partitions only; the `test` partition is never opened.
[configs/ml_cnn_baseline.json](configs/ml_cnn_baseline.json) supplies the model
and optimization settings: batch size 32 samples (matrix groups are never
split across batches), up to 500 epochs, Adam at learning rate $10^{-3}$, early
stopping after 40 epochs without improvement, seed 2026, deterministic
algorithms enabled.

Smoke run — one epoch, enough to verify the data path and checkpoint writing,
and **not** a trained model:

```bash
python scripts/train_rho_cnn.py \
  --config configs/ml_cnn_baseline.json \
  --epoch-limit 1
```

Full training (expensive):

```bash
python scripts/train_rho_cnn.py --config configs/ml_cnn_baseline.json
```

Resume exactly from the last checkpoint, restoring optimizer and RNG state.
Required after an interrupted run, and the way to continue an
`--epoch-limit`-bounded run:

```bash
python scripts/train_rho_cnn.py \
  --config configs/ml_cnn_baseline.json \
  --resume
```

`--device` overrides the configured device and accepts `auto`, `cpu`, `cuda`,
or `mps`. The output directory named in the config (`weights/cnn_paper_v1`)
receives `latest.pt`, `best.pt`, `history.csv`, and `summary.json`; the
training report is written to `reports/ml_phase_3_training_report.md`. Starting
a fresh run into a non-empty output directory is refused — use `--resume` or a
new directory.

### D. Evaluate on the held-out split

[scripts/evaluate_rho_cnn.py](scripts/evaluate_rho_cnn.py) runs the locked
held-out test evaluation described by
[configs/ml_cnn_evaluation.json](configs/ml_cnn_evaluation.json):

```bash
python scripts/evaluate_rho_cnn.py --config configs/ml_cnn_evaluation.json
```

The evaluation is locked: the first run writes `evaluation_lock.json` recording
the checkpoint checksum, the sample-index and split checksums, the sample and
matrix counts, and a per-artifact digest. A later invocation verifies that lock
against the current sources and reuses the stored result without running the
model again. Mismatched sources are an error rather than a silent re-evaluation,
so a new evaluation requires a new output directory.

Reported metrics:

- **Regression**, overall and stratified by mesh family, level, pattern,
  $\varepsilon$, and $\theta$: MSE, RMSE, MAE, bias, $R^2$, Pearson
  correlation, maximum absolute error, absolute-error quantiles (p50/p90/p95/p99),
  and counts of predictions outside $[0, 1]$.
- **Uncertainty**: bootstrap confidence intervals resampled over matrix
  checksums, 2000 replicates at the 95% level with seed 2026.
- **$\theta$ selection**, per matrix: the exact-optimum rate (how often the
  argmin of predicted $\rho$ is a true argmin), and the mean, median, 95th
  percentile, and maximum of the $\rho$ regret incurred by the selected
  $\theta$, both overall and per mesh family.
- The RMSE a constant train-mean predictor would achieve, as a baseline.

Artifacts are written to `results/cnn_paper_v1/test_v1` (`metrics.json`,
`predictions.jsonl` / `.csv`, `theta_decisions.jsonl` / `.csv`,
`evaluation_lock.json`, plus PNG plots when Matplotlib is available), and the
report to `reports/ml_phase_4_test_report.md`.

### E. Predict $\rho$ and recommend $\theta$ for one matrix

[scripts/predict_rho.py](scripts/predict_rho.py) scores one finalized CSR
operator — any `.npz` produced by the experiment runner — against a grid of
candidate thresholds:

```bash
MATRIX_PATH=datasets/medium/simplex/matrices/simplex_l5_checkerboard_4x4_e2_high_white.npz

python scripts/predict_rho.py \
  --config configs/ml_cnn_inference.json \
  --matrix "$MATRIX_PATH" \
  --output results/prediction.json
```

Candidates default to the 13 values in `default_theta_values` of
[configs/ml_cnn_inference.json](configs/ml_cnn_inference.json) and can be
replaced with `--theta-values 0.1,0.24,0.5` or repeated `--theta 0.24` options
(the two are mutually exclusive). The refinement level is read from the
identity embedded in the `.npz`; pass `--level N` when it is absent, and note
that an explicit value conflicting with the embedded one is rejected.
`--device` accepts `auto`, `cpu`, `cuda`, `mps`.

The JSON result contains the matrix identity and checksum, the candidate list,
one `predicted_rho` per candidate, the `recommendation` object holding the
selected `theta` and its `predicted_rho`, the level/$\theta$ support of the
training data with warnings when the request falls outside it, feature and
inference timings, and model provenance. Without `--output` the document goes
to stdout and the one-line summary to stderr.

### F. End-to-end ANN-selected AMG solve

[scripts/run_predicted_amg.py](scripts/run_predicted_amg.py) chains the whole
loop: assemble the operator with the C++ driver, convert it to CSR, ask the CNN
for a $\theta$, run BoomerAMG at that threshold, and compare predicted against
measured convergence.

```bash
python scripts/run_predicted_amg.py \
  --inference-config configs/ml_cnn_inference.json \
  --build-dir build-unified \
  --output-dir results/predicted_amg/simplex_l5_checkerboard_4x4_e2 \
  --mesh-family simplex \
  --level 5 \
  --pattern checkerboard_4x4 \
  --epsilon 2 \
  --device auto
```

`--mesh-family` accepts `simplex` or `polygonal`; the polygonal case needs
`meshaware_diffusion_polydeal` in the build directory. Solver behaviour is
tunable through `--high-region`, `--rtol`, `--atol`, `--max-iterations`,
`--repeats`, and `--warmup-runs`, and the candidate grid through
`--theta-values`.

The command prints `selected_theta`, `predicted_rho`, and `measured_rho`, and
writes `operator.npz`, `recommendation.json`, `records/*.json`, `manifest.json`,
and `workflow_lock.json` into `--output-dir`. The output directory is staged
and moved into place atomically; if it already exists, the stored lock is
verified against the current request and the previous result is returned
unchanged rather than recomputed.

## Configuration guide

| File | Purpose |
| --- | --- |
| [configs/common.json](configs/common.json) | Shared defaults merged into every experiment config in the same directory: patterns, $\varepsilon$ grid, high region, CG tolerances, AMG backend/profile/smoother, warm-ups, matrix format |
| [configs/medium.json](configs/medium.json) | Primary sweep — three mesh families, levels 3/5/8/10, 10 $\theta$ values |
| [configs/small.json](configs/small.json) | Reduced sweep — levels 3/4/6, 5 $\theta$ values; the second dataset tier the ML pipeline reads |
| [configs/smoke.json](configs/smoke.json) | Single quadrilateral trial; the runner's default config |
| [configs/medium_simplex_dg.json](configs/medium_simplex_dg.json) | Simplex SIPG sweep with flat (non-per-family) output |
| [configs/medium_polygonal_boomeramg_nodal.json](configs/medium_polygonal_boomeramg_nodal.json) | Polygonal sweep using the `polygonal-nodal` BoomerAMG profile |
| [configs/medium_polygonal_chebyshev.json](configs/medium_polygonal_chebyshev.json) | Polygonal sweep using the explicit PolyDeal agglomeration hierarchy |
| [configs/ml_cnn_baseline.json](configs/ml_cnn_baseline.json) | CNN architecture, optimization schedule, dataset index paths, checkpoint and report destinations |
| [configs/ml_cnn_evaluation.json](configs/ml_cnn_evaluation.json) | Held-out evaluation: training config to reuse, checkpoint, bootstrap settings, output directory |
| [configs/ml_cnn_inference.json](configs/ml_cnn_inference.json) | Inference contract: training config, checkpoint, training summary, default candidate $\theta$ values |

Remaining files in [configs/](configs/) are narrower variants of the same
schemas (smoke tests, convergence studies, pilots, and the reference-table
grid).

## Repository structure

| Path | Responsibility |
| --- | --- |
| [include/meshaware/](include/meshaware/) | Header-only core: coefficient patterns and the manufactured solution, the trial-record schema, the PETSc/BoomerAMG solver interface, and shared CLI parsing helpers |
| [src/common/](src/common/) | BoomerAMG configuration, CG solve and metric extraction, PETSc matrix export, JSON record writing |
| [src/dealii/](src/dealii/) | deal.II drivers: continuous $Q_1$/$P_1$ (`diffusion_dealii.cpp`) and simplex SIPG (`diffusion_simplex_dg.cpp`) |
| [src/polydeal/](src/polydeal/) | Polygonal agglomerated-DG driver, including the nodal BoomerAMG profile, the PolyDeal agglomeration multigrid backend, and the direct-solve oracle |
| [python/meshaware_data/](python/meshaware_data/) | Experiment-config schema and grid expansion, solver command construction, stable matrix/sample identifiers, PETSc→CSR conversion and validation, atomic artifact writing, per-family CSV reporting |
| [python/meshaware_ml/](python/meshaware_ml/) | Dataset snapshots and feature cache, matrix pooling, leakage-safe indexing and splitting, PyTorch dataset and model, training, locked evaluation, inference, and the end-to-end AMG integration |
| [scripts/](scripts/) | Command-line entry points for sweeps, ML pipeline construction, training, evaluation, inference, the predicted-AMG workflow, and figure/table generation |
| [configs/](configs/) | Versioned JSON experiment and ML configurations |

Generated content — datasets, ML indices, checkpoints, evaluation results, and
reports — is written to `datasets/`, `weights/`, `results/`, and `reports/`.
These are not tracked in the repository; a fresh clone creates them on first
use.

## Scientific outputs and reproducibility

**Trial records.** Every solve writes a JSON record carrying the mesh identity
($\text{family}, \text{level}, h_{\text{nominal}}, h_{\max}$), the coefficient
setting (pattern, $\varepsilon$, high region), the AMG configuration (backend,
profile, smoother, relaxation weight, $\theta$), problem size (cells, DoFs,
nnz), solver outcome (CG iterations, converged reason, initial and final
residual), AMG hierarchy data (levels, grid and operator complexity),
discretization errors ($L^2$, $H^1$ seminorm, energy), and the path and format
of the stored operator.

**Timing separation.** Assembly, AMG setup, and solve are timed independently
and stored as separate fields, so preconditioner construction cost can be
attributed apart from the Krylov iteration. Warm-up runs are executed and
discarded before the measured repeats.

**Convergence factor.** $\rho$ is computed from the residual history as
$(\|r_{\text{final}}\| / \|r_{\text{initial}}\|)^{1/k}$ over the $k$ CG
iterations actually performed. This is the quantity recorded per trial and the
quantity the CNN regresses.

**Matrix artifacts.** Operators are exported in PETSc binary form, converted to
compressed CSR `.npz` with an embedded identity block, validated against the
shape and nnz recorded in the trial record, and only then does the runner delete
the PETSc staging file and rewrite the record's `matrix_path`. Reused matrices
are revalidated rather than regenerated.

**Leakage-safe splits.** Train/validation/test membership is decided per matrix
content checksum, not per sample, so no operator can be seen during training
and scored at test time. The dataset loader asserts hash disjointness between
partitions at construction time.

**Deterministic training.** Seeds are applied to Python, NumPy, and PyTorch,
deterministic algorithms are requested, and checkpoints store optimizer and RNG
state so `--resume` continues a run exactly rather than approximately.

**Locked evaluation.** The held-out evaluation records checksums of the
checkpoint, sample index, and split file, along with the exact sample and matrix
counts and `model_updates: 0`. Re-running verifies that lock instead of scoring
the test split again.

## Citation and license

This work reproduces and extends the heterogeneous diffusion experiment of:

> P. F. Antonietti, M. Caldana, L. Dede'. *Accelerating Algebraic Multigrid
> Methods via Artificial Neural Networks*.
> arXiv:[2111.01629](https://arxiv.org/abs/2111.01629),
> DOI [10.1007/s10013-022-00597-w](https://doi.org/10.1007/s10013-022-00597-w)

The polygonal driver builds on PolyDeal, whose authors ask that you
cite:

> M. Feder, A. Cangiani, L. Heltai. *R3MG: R-tree based agglomeration of
> polytopal grids with applications to multilevel methods*. Journal of
> Computational Physics, 526 (2025).
> <https://doi.org/10.1016/j.jcp.2025.113773>

No license file is currently provided in this repository, so no license terms
are granted.
