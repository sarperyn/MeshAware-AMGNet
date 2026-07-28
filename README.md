# MeshAware-AMG

Reproducible finite-element experiments for learning how HYPRE BoomerAMG's
strong-threshold parameter affects heterogeneous diffusion problems.

The implementation target is the experiment in Section 5 of
arXiv:2111.01629v2:

\[
-\nabla\cdot(\mu\nabla u)=f\quad\text{in }(-1,1)^2,
\]

with four tile patterns, twelve coefficient contrasts, eight mesh sizes, and a
CG solver preconditioned by BoomerAMG. Quadrilateral and simplex discretizations
use deal.II; polygonal SIPG discretizations use PolyDeal.

## Current vertical slice

- Versioned experiment configurations in `configs/`.
- Dependency-free C++ definitions of the four coefficient fields and exact
  solutions.
- A shared serial deal.II quadrilateral Q1/simplex P1 driver using PETSc CG +
  HYPRE BoomerAMG.
- A PolyDeal SIPG polygonal driver with production PETSc assembly and an
  optional native direct-solve equivalence oracle.
- Lossless compressed CSR NPZ storage and one JSON record per solver trial.
- Separate assembly, AMG setup, and solve timings.
- Assemble-once matrix batches over all theta values and timing repeats, with
  configurable discarded warm-ups.
- Storage/resource preflight and validated atomic PETSc-to-CSR-NPZ conversion.
- Python configuration expansion/validation and unit tests.

The paper's exact cosine solution has nonzero Dirichlet values on parts of
`boundary (-1,1)^2`; the implementation therefore applies the exact trace,
not homogeneous boundary data. At epsilon zero this is the uniform Poisson
baseline.

## Configure and test

The core tests do not require deal.II:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s tests/python -v
```

If a PETSc-enabled deal.II 9.7 or newer is discoverable, CMake also builds
`meshaware_diffusion_dealii`. Otherwise it prints a status message and leaves the
dependency-free validation targets available. Set `DEAL_II_DIR` when deal.II is
installed in a nonstandard location.

For the local unified installation used by this repository:

```bash
cmake -S . -B build-unified \
  -DDEAL_II_DIR=/Users/sarperyurtseven/local/dealii-9.7.1-unified/lib/cmake/deal.II \
  -DPOLYDEAL_ROOT=/Users/sarperyurtseven/local/Polydeal \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-unified --parallel
ctest --test-dir build-unified --output-on-failure
```

`meshaware_diffusion_polydeal` builds deterministic L-shaped polyominoes inside
a common 4-by-4 interface layout, so the polygon mesh is identical across
patterns and no polygon crosses any coefficient discontinuity. Production mode
assembles only into PETSc, uses the shared CG/BoomerAMG timing path, exports the
sparse matrix, and writes the common trial schema. It reports L2, broken-H1,
and SIPG energy errors. Passing `--oracle` additionally assembles the native
matrix, requires entrywise matrix/RHS agreement, and compares UMFPACK with the
iterative solution at moderate contrast.

Polygonal validation can be run with:

```bash
python3 scripts/run_experiments.py \
  --config configs/polygonal_convergence.json \
  --build-dir build-unified \
  --output-root datasets \
  --overwrite-records
python3 scripts/check_convergence.py \
  --records-glob 'datasets/polygonal_convergence/**/records/*.json'
```

The orchestrator launches one finite-element process per matrix. That process
assembles and optionally exports the matrix once, then runs every configured
theta and repeat with a fresh KSP/AMG hierarchy. By default, one solve per theta
is discarded before measurements begin. Existing records are skipped on a
restart, and a completely finished matrix batch does not launch a process.
Every trial record includes the resulting BoomerAMG hierarchy depth as
`amg_levels`; generated report CSVs expose the same quantity as `n_levels`.

Validate this execution path across all three families with:

```bash
python3 scripts/run_experiments.py \
  --config configs/batch_smoke.json \
  --build-dir build-unified \
  --output-root datasets
```

`--limit` intentionally uses the older one-process-per-trial path for quick
debugging of an arbitrary prefix. Do not use it for production timing.

Generate the paper-style figures with the scientific plotting environment:

```bash
python3 -m venv .venv-figures
.venv-figures/bin/python -m pip install -r requirements-figures.txt
PYTHON=.venv-figures/bin/python bash scripts/run_figure_pipeline.sh \
  --dataset-root datasets/small \
  --output-root results/figures/small
```

The theta-versus-levels figure uses statsmodels OLS statistics and LOWESS,
seaborn KDE/histograms, and matplotlib output. It writes both PNG and SVG for
each mesh family.

Before generating a tier, inspect its storage and largest-matrix estimate:

```bash
python3 scripts/storage_preflight.py \
  --config configs/small.json \
  --json-output storage/small.json
```

Matrix generation retains one lossless compressed CSR NPZ per matrix. The
runner converts and validates each PETSc staging file immediately after its
theta/repeat batch, updates all records, and only then removes the staging
file. NumPy is therefore required by the experiment runner:

```bash
python3 -m venv .venv-data
.venv-data/bin/python -m pip install -r requirements-data.txt
```

Small is safe on the current disk. Compressed storage makes the medium disk
estimate practical, but its level-10 solve still requires the documented
memory pilot; see [storage and conversion](docs/storage_and_conversion.md).

Example solver invocation:

```bash
./build/meshaware_diffusion_dealii \
  --mesh-family quadrilateral --level 3 --epsilon 0 \
  --pattern vertical_split --theta 0.24 \
  --record results/smoke.json --matrix results/smoke.petsc
```

The corresponding direct batch interface is:

```bash
./build-unified/meshaware_diffusion_dealii \
  --mesh-family quadrilateral --level 3 --epsilon 0 \
  --pattern vertical_split --theta-values 0.2,0.4 \
  --repeats 2 --warmup-runs 1 \
  --record-dir results/records --matrix results/matrix.petsc
```

Run the numerical convergence gate for both conforming mesh families with:

```bash
python3 scripts/run_experiments.py \
  --config configs/convergence.json --build-dir build-unified \
  --output-root datasets --overwrite-records
python3 scripts/check_convergence.py
```

See [the source tree](docs/source_tree.md), [experiment
protocol](docs/experiment_protocol.md), [timing protocol](docs/timing_protocol.md),
[dataset schema](docs/dataset_schema.md), and [storage/conversion
protocol](docs/storage_and_conversion.md) before producing dataset results.
