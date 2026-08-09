# AMG-ANN Repository Instructions

## Project purpose

This repository implements AMG experiments and artificial-neural-network methods for predicting or selecting AMG parameters. It may contain C++, Python, CMake, deal.II, Hypre, PyTorch, experiment scripts, configuration files, trained models, and legacy implementations.

The repository has evolved through many overlapping experiments. The current goal is to remove obsolete code, reduce duplication, simplify the architecture, and preserve every supported workflow.

## Python environment

Use the Python executable stored in `AMG_PYTHON`.

Before running any Python command, verify it:

```bash
test -n "$AMG_PYTHON"
test -x "$AMG_PYTHON"
"$AMG_PYTHON" -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

Use:

```bash
"$AMG_PYTHON" script.py
"$AMG_PYTHON" -m pytest
"$AMG_PYTHON" -m pip
"$AMG_PYTHON" -m ruff
```

Never use bare `python`, `python3`, `pip`, or `pytest`.

Do not create another environment or install or change packages unless the task requires it and the user approves.

## Cleanup invariants

Preserve:

* Supported AMG solver behavior.
* ANN training and inference.
* Model and checkpoint compatibility when reasonably possible.
* Configuration-driven execution paths.
* deal.II, Hypre, mesh, and solver variants that are still supported.
* Numerical correctness and reproducibility.
* Existing user changes in the working tree.

Do not delete something solely because a textual search finds no caller.

Before declaring code unused, inspect:

* Direct and indirect call sites.
* Imports and dynamic imports.
* CMake targets and generated code.
* Configuration files and command-line options.
* Registries, callbacks, factories, and templates.
* Experiment and evaluation scripts.
* Checkpoint and model-loading compatibility.
* Tests, examples, documentation, and notebooks.

## Cleanup workflow

For substantial cleanup:

1. Inspect `git status` and preserve pre-existing changes.
2. Discover the repository structure, entry points, build commands, tests, configurations, and supported workflows.
3. Establish a working baseline before editing.
4. Run representative AMG and ANN smoke tests and record their important output.
5. Build an evidence-backed list of cleanup candidates.
6. Classify each candidate as:

   * Safe to remove.
   * Duplicate and suitable for consolidation.
   * Obsolete but compatibility-sensitive.
   * Uncertain and therefore retained.
7. Make changes in small, coherent batches.
8. After each batch, run targeted validation and inspect the diff.
9. Continue until no other evidence-supported cleanup remains.
10. Report retained ambiguities instead of guessing.

## Validation

Use the repository's documented commands when available.

For Python changes, run relevant combinations of:

```bash
"$AMG_PYTHON" -m pytest
"$AMG_PYTHON" -m ruff check .
```

For C++ changes, run the relevant CMake build and CTest targets.

Also run at least one representative end-to-end AMG-ANN workflow when practical.

Compare important numerical results such as:

* Solver convergence.
* Final residual.
* CG iteration count.
* AMG setup behavior.
* Predicted strong-threshold value.
* ANN output.
* Experiment result structure.

Small floating-point differences are acceptable only when explained and within an appropriate tolerance.

## Refactoring rules

Prefer:

* Deleting proven dead code.
* One clear implementation per concept.
* Explicit module responsibilities.
* Small functions with descriptive names.
* Shared implementations instead of copy-pasted variants.
* Configuration objects instead of long parameter lists.
* Comments that explain mathematical intent or non-obvious constraints.

Do not:

* Perform a wholesale rewrite.
* Introduce unnecessary abstractions.
* Shorten code by hiding important numerical logic.
* Silently change defaults or experiment semantics.
* Mix unrelated cleanup into one change.
* Replace working project code with speculative generated architecture.

Code quality is measured by clarity and maintainability, not only by line count.

## Git behavior

Do not reset, discard, or overwrite pre-existing user changes.

Do not commit, push, or open a pull request unless explicitly requested.

At completion, report:

* Files changed.
* Code removed.
* Duplication consolidated.
* Behavior intentionally preserved.
* Tests and experiments run.
* Failures or unavailable validation.
* Ambiguous components retained and why.
