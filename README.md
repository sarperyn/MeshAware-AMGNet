# MeshAware-AMG

Reproducible finite-element experiments for heterogeneous diffusion,
polygon-aware multigrid preconditioners, and machine-learning tools for
selecting the HYPRE BoomerAMG strong-threshold parameter.

## Overview

MeshAware-AMG studies how the strong-connection threshold \(\theta\) affects
algebraic multigrid performance for finite-element systems with discontinuous
diffusion coefficients. It reproduces and extends the heterogeneous diffusion
experiment from [*Accelerating Algebraic Multigrid Methods via Artificial
Neural Networks*](https://arxiv.org/abs/2111.01629).

The model problem is

\[
-\nabla \cdot \left(\mu(x,y)\nabla u\right)=f
\qquad \text{in } \Omega=(-1,1)^2.
\]

The repository covers four coefficient patterns, twelve coefficient contrasts,
multiple refinement levels, and three mesh families:

- Quadrilateral Q1 elements.
- Simplex P1 elements.
- Polygonal symmetric interior-penalty discontinuous Galerkin elements.

The C++ solvers assemble finite-element operators and run conjugate gradient
with a multigrid preconditioner. Quadrilateral and simplex systems use HYPRE
BoomerAMG. Polygonal systems can use default scalar BoomerAMG, a block-aware
nodal BoomerAMG profile, or an explicit PolyDeal agglomeration hierarchy. The
Python pipeline manages experiments, validates and stores sparse matrices,
produces statistical reports, trains a convolutional model, and evaluates
model-selected BoomerAMG parameters.

## Scientific scope

The quadrilateral and simplex implementations in [src/dealii/](src/dealii/)
use [deal.II](https://dealii.org/). The polygonal implementation in
[src/polydeal/](src/polydeal/) uses
[PolyDeal](https://github.com/fdrmrc/Polydeal) and a symmetric interior-penalty
discontinuous Galerkin formulation.

The BoomerAMG experiments use the same [PETSc](https://petsc.org/)
conjugate-gradient and
[HYPRE BoomerAMG](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html)
measurement path. This keeps solver configuration, residual handling, timing
boundaries, and output records consistent across discretizations. The explicit
PolyDeal hierarchy uses deal.II CG with Trilinos matrices but follows the same
residual, timing, and record contract.

The manufactured cosine solution supplies the forcing term and the full
Dirichlet trace. At \(\epsilon=0\), the coefficient is uniform and the problem
becomes the Poisson baseline. For heterogeneous cases, \(\mu\) takes the values
\(1\) and \(10^\epsilon\). The selected high-coefficient region is recorded
explicitly because the reference paper contains two conflicting color
conventions.

## Solver and preconditioner variants

The repository currently implements the following solver paths:

| Discretization | Preconditioner | Selection | Hierarchy construction |
|---|---|---|---|
| Quadrilateral Q1 or simplex P1 | HYPRE BoomerAMG | Default | Scalar algebraic strength, coarsening, and interpolation |
| Polygonal SIPG | HYPRE BoomerAMG | `--amg-backend boomeramg --boomeramg-profile default` | Scalar algebraic hierarchy |
| Polygonal SIPG | HYPRE nodal BoomerAMG | `--amg-backend boomeramg --boomeramg-profile polygonal-nodal` | Block-aware algebraic hierarchy using one three-DoF node per polygon |
| Polygonal SIPG | PolyDeal agglomeration multigrid | `--amg-backend polydeal-agglomeration` | Explicit nested polygon agglomerates and Galerkin coarse operators |

The `polygonal-nodal` option is still an **algebraic multigrid
preconditioner**. It does not construct a geometric mesh hierarchy. It gives
BoomerAMG finite-element metadata that the scalar matrix graph does not
contain: the three modal `FE_AggloDGP(1)` coefficients on one polygon form a
block, and the projected constant function is a near-nullspace mode. HYPRE
then builds the hierarchy algebraically using nodal coarsening, HMIS, and
extended+i interpolation.

The explicit `polydeal-agglomeration` backend is a separate implementation. It
merges neighboring polygons without crossing the common coefficient-tile
boundaries, builds modal transfer operators and Galerkin level matrices, uses
a direct coarse solve, and applies a symmetric V-cycle preconditioner to CG.
The BoomerAMG strength threshold `theta` does not apply to this backend.

## Techniques worth studying

- **Shared mathematical definitions.** Coefficient fields, exact solutions,
  gradients, and forcing terms are defined once in
  [include/meshaware/](include/meshaware/). Both finite-element drivers use the
  same definitions.

- **Comparable conforming and polygonal discretizations.** Quadrilateral Q1
  and simplex P1 systems share one deal.II driver. The PolyDeal path uses SIPG
  with one-sided diffusion values and a coefficient-scaled penalty.

- **Interface-aligned polygonal meshes.** Deterministic L-shaped polyominoes
  are constructed inside a common \(4\times4\) tile layout. No polygon crosses
  a possible coefficient discontinuity, and geometry remains fixed when the
  coefficient pattern changes.

- **Block-aware algebraic multigrid for polygonal DG.** The nodal BoomerAMG
  profile groups the three degree-one modal unknowns on each polygon and
  supplies the correctly projected constant mode. This prevents scalar
  coarsening from splitting a physical polygon and improves the coarse
  representation of low-energy error.

- **Explicit polygon agglomeration hierarchy.** The alternative PolyDeal
  backend recursively merges polygons inside coefficient tiles, constructs
  exact modal injection operators, obtains coarse matrices by Galerkin
  projection, and applies configurable five-step smoothers in a V-cycle.

- **Independent solver timings.** Assembly, AMG hierarchy construction, and
  iterative solution are measured separately. Calling `KSPSetUp` before
  `KSPSolve` prevents lazy preconditioner construction from entering the solve
  interval.

- **Assemble-once experiment batches.** Each matrix is assembled and exported
  once. Every BoomerAMG threshold and timing repeat receives a fresh PETSc KSP
  and hierarchy, including discarded warm-up runs. The explicit PolyDeal
  backend similarly rebuilds its multigrid state for every timing run.

- **Residual-based convergence measurements.** The solver records the
  unpreconditioned residual history and computes
  \[
  \rho=\left(\frac{\lVert r_N\rVert_2}{\lVert r_0\rVert_2}\right)^{1/N}.
  \]
  It also records hierarchy depth, grid complexity, operator complexity, and
  the solver convergence reason.

- **Independent polygonal correctness oracle.** Validation mode inserts the
  same SIPG contributions into native and PETSc matrices, checks entrywise
  agreement, verifies symmetry, and compares the iterative solution with
  [SuiteSparse UMFPACK](https://github.com/DrTimothyAldenDavis/SuiteSparse).

- **Bounded-memory sparse conversion.** The data layer in
  [python/meshaware_data/](python/meshaware_data/) reads PETSc binary matrices
  through [NumPy memory
  mapping](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html),
  validates their structure in chunks, and converts them to compressed CSR NPZ
  artifacts without densification.

- **Transactional artifact publication.** Matrices, records, features,
  checkpoints, and evaluation outputs are written to temporary paths,
  validated, and atomically promoted. SHA-256 fingerprints connect derived
  data to its exact source.

- **Sparse matrix image pooling.** The learning pipeline in
  [python/meshaware_ml/](python/meshaware_ml/) converts each sparse operator
  into a three-channel \(100\times100\) view using positive maximum, negative
  maximum, and sum reductions. Pooling operates directly on CSR arrays and
  avoids a dense copy of the original matrix.

- **Leakage-safe data partitions.** Samples derived from the same matrix
  checksum stay in one train, validation, or test partition. Existing
  assignments remain stable when new matrices are added.

- **Conditioned convergence prediction.** A
  [PyTorch](https://pytorch.org/) convolutional network combines the pooled
  operator embedding with mesh scale and candidate \(\theta\), then predicts
  the AMG convergence factor.

- **Reproducible training and locked evaluation.** Training captures
  random-number-generator state, supports exact checkpoint continuation, and
  uses early stopping. Held-out evaluation includes stratified errors,
  per-matrix threshold regret, and grouped-bootstrap confidence intervals.

- **Multiple scientific figure backends.** Reporting code produces raster
  figures with [Matplotlib](https://matplotlib.org/), publication-oriented TeX
  with [PGFPlots](https://ctan.org/pkg/pgfplots), and direct
  [SVG](https://developer.mozilla.org/en-US/docs/Web/SVG) output.

## Current checked-in results

### Polygonal multigrid experiments

The original medium polygonal sweep exposed severe mesh-dependent iteration
growth with scalar BoomerAMG. The difficult level-8, `checkerboard_4x4`,
\(\epsilon=7\) cases required 2,232--2,703 CG iterations across the theta
grid. With the nodal BoomerAMG profile, the same matrices require 116--278
iterations. At the best observed nodal theta, the convergence factor falls
from approximately 0.992 to 0.852.

Across the 1,519 matching records currently available in both datasets, the
median iteration count falls from 154 to 71, and 1,497 records improve. The
level summary is:

| Refinement | Scalar BoomerAMG median / maximum | Nodal BoomerAMG median / maximum |
|---|---:|---:|
| 3 | 62.5 / 254 | 46 / 207 |
| 5 | 144 / 809 | 81.5 / 568 |
| 8 | 308 / 2,703 | 98.5 / 776 |
| 10, partial | 425 / 447 | 75 / 178 |

Levels 3, 5, and 8 are complete in both compared sweeps; level 10 remains
partial. These numbers measure the combined nodal profile, not an ablation of
its individual HYPRE settings. See the
[scalar trials](datasets/medium/polygonal/diffusion_reports/trials.csv),
[nodal trials](datasets/medium-polygonal-boomeramg-nodal/polygonal/diffusion_reports/trials.csv),
and [nodal configuration](configs/medium_polygonal_boomeramg_nodal.json).

The checked-in PolyDeal agglomeration experiment contains all 192 configured
matrices across levels 3, 5, 8, and 10 with a Chebyshev-smoothed V-cycle. It is
a backend comparison rather than a theta sweep because that hierarchy does not
use BoomerAMG's strength threshold. See
[the configuration](configs/medium_polygonal_chebyshev.json) and
[its trial table](datasets/medium/polygonal-chebyshev/diffusion_reports/trials.csv).

### Machine-learning pipeline

The frozen `cnn_paper_v1` dataset contains 4,808 samples from simplex and
polygonal **default BoomerAMG** measurements. The selected checkpoint trained
on 4,098 samples, used 236 validation samples, and was evaluated once on 474
held-out samples grouped by source-matrix hash. The locked test result is
RMSE 0.0390, \(R^2=0.9907\), and mean selected-theta convergence-factor regret
0.00513. Exact best-grid-point selection is 19.64%, so the model should be
treated as an approximate convergence-ranking policy rather than an exact
theta classifier.

The current checkpoint was not trained on nodal BoomerAMG or the explicit
PolyDeal hierarchy; mixing those records into this model version would assign
different convergence targets to the same matrix and theta. The end-to-end
inference bridge has nevertheless been exercised successfully on supported
simplex and polygonal default-BoomerAMG smoke cases. See the
[training report](reports/ml_phase_3_training_report.md),
[locked test report](reports/ml_phase_4_test_report.md), and
[inference integration report](reports/ml_phase_5_inference_amg_report.md).

## End-to-end workflow

Run the following steps from the repository root. The examples use
`build-unified` for compiled programs, `datasets/` for numerical results,
`results/` for derived outputs, and `weights/` for trained models.

### 1. Prepare the toolchain

The C++ code requires:

- CMake 3.20 or newer and a C++17 compiler.
- deal.II 9.7 or newer, built with PETSc and HYPRE support.
- PolyDeal built against the same deal.II installation when polygonal
  experiments are required.
- Trilinos support in deal.II when building the polygonal driver; the explicit
  agglomeration backend stores its hierarchy in Trilinos sparse matrices.

The Python code uses Python 3.10 or newer. Create an environment containing the
numerical, learning, and plotting dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy torch pandas matplotlib seaborn statsmodels \
  scikit-learn
```

`pdflatex` and PGFPlots are optional. They are needed only for PDF figure
generation; the plotting pipeline can still produce CSV, PNG, and SVG outputs
without them.

### 2. Configure, build, and test the solvers

Point CMake to the deal.II and PolyDeal installations on the local machine:

```bash
cmake -S . -B build-unified \
  -DDEAL_II_DIR=/path/to/deal.II/lib/cmake/deal.II \
  -DPOLYDEAL_ROOT=/path/to/Polydeal \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-unified --parallel
ctest --test-dir build-unified --output-on-failure
python -m unittest discover -s tests/python -v
```

The build should produce `meshaware_diffusion_dealii` for quadrilateral and
simplex problems and `meshaware_diffusion_polydeal` for polygonal problems.
If only deal.II is available, disable the polygonal driver with
`-DMESHAWARE_BUILD_POLYDEAL=OFF`.

BoomerAMG supports four smoother selections through the shared solver
interface. The default preserves HYPRE's symmetric SOR/Jacobi relaxation:

```bash
./build-unified/meshaware_diffusion_dealii \
  --mesh-family quadrilateral --level 3 --epsilon 1.2 \
  --pattern vertical_split --theta 0.24 \
  --amg-smoother symmetric-gauss-seidel

./build-unified/meshaware_diffusion_dealii \
  --mesh-family quadrilateral --level 3 --epsilon 1.2 \
  --pattern vertical_split --theta 0.24 \
  --amg-smoother damped-jacobi --jacobi-damping 0.6666666666666666

./build-unified/meshaware_diffusion_dealii \
  --mesh-family quadrilateral --level 3 --epsilon 1.2 \
  --pattern vertical_split --theta 0.24 \
  --amg-smoother chebyshev
```

For experiment sweeps, set `amg_smoother` and `jacobi_damping` in the JSON
configuration. Trial records retain both the smoother name and its effective
relaxation weight. BoomerAMG additionally accepts
`l1-symmetric-gauss-seidel`; it is not available in the explicit PolyDeal
backend and is intentionally rejected by the polygonal nodal profile.

Run the block-aware algebraic profile directly with:

```bash
./build-unified/meshaware_diffusion_polydeal \
  --mesh-family polygonal --level 8 --epsilon 7 \
  --pattern checkerboard_4x4 --theta 0.7044444444444445 \
  --amg-backend boomeramg \
  --boomeramg-profile polygonal-nodal \
  --amg-smoother symmetric-gauss-seidel
```

This remains PETSc CG preconditioned by HYPRE BoomerAMG. The profile changes
how BoomerAMG constructs its algebraic hierarchy; it does not replace AMG with
a geometric solver.

The polygonal driver also provides PolyDeal's agglomeration multigrid
hierarchy:

```bash
./build-unified/meshaware_diffusion_polydeal \
  --mesh-family polygonal --level 6 --epsilon 7 \
  --pattern checkerboard_4x4 \
  --amg-backend polydeal-agglomeration
```

This backend keeps the existing L-shaped polygons as its finest level, merges
nested neighboring polygons without crossing the common 4-by-4 coefficient
tile boundaries, constructs Galerkin coarse operators, and applies a
configurable smoothed V-cycle. It accepts the same `--amg-smoother` choices:
Chebyshev, damped Jacobi, and symmetric Gauss-Seidel (implemented by the
Trilinos SSOR preconditioner). The polygonal experiment configurations use
Chebyshev by default. Set
`"amg_backend": "polydeal-agglomeration"` in a polygonal-only experiment
configuration. `theta` is retained in records for schema compatibility but is
not used by this backend.

### 3. Run a cross-family smoke experiment

The [batch smoke configuration](configs/batch_smoke.json) exercises all three
mesh families, two threshold values, repeated solves, matrix conversion, and
report generation:

```bash
python scripts/run_experiments.py \
  --config configs/batch_smoke.json \
  --build-dir build-unified \
  --output-root datasets
```

Results are written under `datasets/batch_smoke/`. Rerunning the command skips
complete records and resumes incomplete matrix batches. Use
`--overwrite-records` only when the existing measurements should be replaced.

### 4. Check numerical convergence

Run the conforming Q1/P1 convergence study and its rate check:

```bash
python scripts/run_experiments.py \
  --config configs/convergence.json \
  --build-dir build-unified \
  --output-root datasets
python scripts/check_convergence.py \
  --records-glob 'datasets/convergence/**/records/*.json'
```

Validate the polygonal SIPG path separately:

```bash
python scripts/run_experiments.py \
  --config configs/polygonal_convergence.json \
  --build-dir build-unified \
  --output-root datasets
python scripts/check_convergence.py \
  --records-glob 'datasets/polygonal_convergence/**/records/*.json'
```

Do not start a large parameter sweep until the solver tests and convergence
checks pass on the target machine.

For the polygonal SIPG matrix, test the block-aware BoomerAMG profile before
starting the medium sweep:

```bash
python scripts/run_experiments.py \
  --config configs/polygonal_boomeramg_nodal_pilot.json \
  --build-dir build-unified \
  --output-root datasets
```

The pilot groups the three modal `FE_AggloDGP(1)` unknowns belonging to each
polygon, supplies the projected constant mode as near-nullspace information,
and uses HYPRE nodal coarsening with HMIS and extended+i interpolation. It
records grid and operator complexity alongside the usual convergence and
timing metrics. Compare those results with the matching cases in
`datasets/medium/polygonal/`.

The profile has also been promoted to the full medium polygonal grid:

```bash
python scripts/run_experiments.py \
  --config configs/medium_polygonal_boomeramg_nodal.json \
  --build-dir build-unified \
  --output-root datasets
```

Run the one-solve-per-matrix PolyDeal agglomeration comparison with:

```bash
python scripts/run_experiments.py \
  --config configs/medium_polygonal_chebyshev.json \
  --build-dir build-unified \
  --output-root datasets/medium
```

The second command reproduces the checked-in
`datasets/medium/polygonal-chebyshev/` layout because that single-family
configuration intentionally uses a flat output directory.

### 5. Estimate storage and inspect an experiment

The [small](configs/small.json), [medium](configs/medium.json), and
[large](configs/large.json) configurations increase mesh coverage, threshold
resolution, and repetition counts. Inspect the expanded grid without running
it:

```bash
python scripts/run_experiments.py \
  --config configs/small.json \
  --build-dir build-unified \
  --output-root datasets \
  --dry-run
```

Estimate retained storage, temporary conversion space, and solver memory
before generation:

```bash
python scripts/storage_preflight.py \
  --config configs/small.json \
  --json-output results/small_storage_preflight.json \
  --enforce-free-space
```

Run the same preflight with `configs/medium.json` or `configs/large.json`
before selecting a larger tier.

These cross-family tiers inherit `boomeramg_profile: default` from
[configs/common.json](configs/common.json), so their polygonal records are the
scalar-BoomerAMG baseline. Use the dedicated nodal configuration above when a
block-aware polygonal BoomerAMG sweep is intended.

### 6. Generate a numerical dataset

The small tier is the practical starting point:

```bash
python scripts/run_experiments.py \
  --config configs/small.json \
  --build-dir build-unified \
  --output-root datasets
```

For each mesh family, the runner creates:

- `records/` with one JSON document per threshold and repeat;
- `matrices/` with one validated compressed CSR NPZ per finite-element
  operator;
- `diffusion_reports/` with tabular trial data;
- `summaries/` with grouped optimal-threshold results.

The matrix is assembled once for each physical problem. Every threshold and
repeat still receives a fresh AMG hierarchy. Existing valid artifacts are
checked and reused when a run is restarted.

### 7. Generate tables and figures

Generate all report families for the small dataset:

```bash
PYTHON=python bash scripts/run_figure_pipeline.sh \
  --dataset-root datasets/small \
  --output-root results/figures/small
```

The pipeline creates threshold-versus-convergence tables, threshold-versus-cost
plots, normalized convergence/time scatter plots, and
threshold-versus-hierarchy-level diagnostics. Outputs are grouped by mesh
family under `results/figures/small/`.

If LaTeX or PGFPlots is unavailable, add `--no-pdf`. The CSV, PNG, and SVG
outputs do not require the TeX backend.

### 8. Build the ANN dataset

Finish or stop numerical generation before freezing an ML snapshot. The
current model is trained only on simplex and polygonal default-BoomerAMG
records; quadrilateral, polygonal-nodal, and PolyDeal-agglomeration records are
intentionally excluded.

Build pooled matrix views, preserve leakage-safe split assignments, and create
the canonical sample index:

```bash
python scripts/build_ml_pipeline.py \
  --dataset-root datasets \
  --output-root datasets/ml \
  --tier small \
  --mesh-family simplex \
  --mesh-family polygonal \
  --generation-status inactive \
  --report reports/ml_phase_1_2_report.md
```

To reproduce the broader paper dataset, repeat `--tier` for both `small` and
`medium`. Do not add nodal or explicit-agglomeration records to
`cnn_paper_v1`: preconditioner choice changes the measured convergence target
even when the sparse matrix and theta are identical. A model that combines
those solvers needs an explicit backend/profile input and a new dataset/model
version. Reruns reuse features identified by their source checksum and retain
existing train, validation, and test assignments. Use `--reset-splits` only
when intentionally creating a new split version.

### 9. Train the convergence-factor model

The [baseline training configuration](configs/ml_cnn_baseline.json) reads the
canonical training and validation partitions and writes checkpoints under
`weights/cnn_paper_v1/`:

```bash
python scripts/train_rho_cnn.py \
  --config configs/ml_cnn_baseline.json
```

Training stops early when validation error no longer improves. It writes
`best.pt`, `latest.pt`, `history.csv`, and `summary.json`. A stopped run can
continue from the exact saved random and optimizer state:

```bash
python scripts/train_rho_cnn.py \
  --config configs/ml_cnn_baseline.json \
  --resume
```

Use `--epoch-limit 1` for a short pipeline check before committing compute time
to a complete run.

### 10. Evaluate the selected checkpoint

After model selection is complete, run the locked held-out evaluation:

```bash
python scripts/evaluate_rho_cnn.py \
  --config configs/ml_cnn_evaluation.json
```

This step writes predictions, overall and stratified metrics, grouped-bootstrap
confidence intervals, threshold-selection regret, plots, and an artifact lock
under `results/cnn_paper_v1/test_v1/`. A later invocation verifies and reuses
the lock instead of evaluating the test set again.

### 11. Recommend a threshold for an existing matrix

Use a finalized simplex or polygonal CSR NPZ matrix and provide the candidate
threshold grid:

```bash
python scripts/predict_rho.py \
  --matrix datasets/small/simplex/matrices/simplex_l4_vertical_split_e1p2_high_white.npz \
  --theta-values 0.2,0.4,0.6 \
  --output results/prediction.json
```

The output contains the predicted convergence factor for every candidate,
their deterministic ranking, the recommended threshold, and model and matrix
provenance. The model does not support quadrilateral matrices because they are
outside its training scope.

### 12. Run BoomerAMG with the model-selected threshold

The complete inference workflow assembles a new operator, converts it to the
model representation, selects a candidate threshold, runs PETSc/HYPRE, and
records the measured convergence factor:

```bash
python scripts/run_predicted_amg.py \
  --build-dir build-unified \
  --output-dir results/amg_inference/simplex_example \
  --mesh-family simplex \
  --level 3 \
  --pattern vertical_split \
  --epsilon 1.2 \
  --theta-values 0.2,0.4,0.6
```

Use `--mesh-family polygonal` for the PolyDeal discretization with the default
BoomerAMG profile. The current inference command does not select or model the
polygonal-nodal or explicit-agglomeration preconditioners. The output directory
contains the operator, recommendation, solver record, workflow manifest, and
SHA-256 lock. Repeating an identical invocation verifies and reuses the locked
artifacts; it does not silently overwrite them.


## Project structure

```text
MeshAware-AMG/
├── CMakeLists.txt
├── README.md
├── configs/
├── datasets/
├── docs/
├── include/
│   └── meshaware/
├── python/
│   ├── meshaware_data/
│   └── meshaware_ml/
├── reports/
├── results/
├── scripts/
├── src/
│   ├── common/
│   ├── dealii/
│   └── polydeal/
├── tests/
├── utils/
└── weights/
```

[configs/](configs/) contains versioned experiment, training, evaluation, and
inference parameters. Values that define a scientific run belong here rather
than in plotting or orchestration code.

[datasets/](datasets/) stores solver records and canonical sparse matrices. Its
[datasets/ml/](datasets/ml/) subtree contains frozen inventories, pooled
features, sample indexes, split assignments, and audit metadata.

[src/common/](src/common/) owns the shared PETSc/HYPRE solver contract and
experiment-record implementation. [src/dealii/](src/dealii/) and
[src/polydeal/](src/polydeal/) contain the conforming and polygonal assembly
paths.

[python/meshaware_data/](python/meshaware_data/) handles schemas, reporting,
storage estimates, and PETSc-to-CSR conversion.
[python/meshaware_ml/](python/meshaware_ml/) handles pooling, dataset indexing,
training, evaluation, inference, and solver integration.

[reports/](reports/) contains human-readable scientific summaries.
[results/](results/) contains figures, evaluation artifacts, and model-selected
AMG runs. [weights/](weights/) contains model checkpoints.

[tests/](tests/) covers mathematical coefficient fields, storage transactions,
schemas, orchestration, pooling, training, evaluation, and inference.
