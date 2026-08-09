#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_directory}/.." && pwd)"
experiment_mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${experiment_mode}" in
  smoke)
    experiment_config="${repository_root}/configs/simplex_dg_smoke.json"
    ;;
  convergence)
    experiment_config="${repository_root}/configs/simplex_dg_convergence.json"
    ;;
  medium)
    experiment_config="${repository_root}/configs/medium_simplex_dg.json"
    ;;
  *)
    echo "usage: $0 {smoke|convergence|medium} [run_experiments options]" >&2
    exit 2
    ;;
esac

if [[ -z "${AMG_PYTHON:-}" ]]; then
  echo "AMG_PYTHON must name the repository Python executable" >&2
  exit 2
fi
if [[ ! -x "${AMG_PYTHON}" ]]; then
  echo "AMG_PYTHON is not executable: ${AMG_PYTHON}" >&2
  exit 2
fi

"${AMG_PYTHON}" -c \
  'import sys, torch; print(sys.executable); print(torch.__version__)'

exec "${AMG_PYTHON}" "${repository_root}/scripts/run_experiments.py" \
  --config "${experiment_config}" \
  --build-dir "${MESHAWARE_BUILD_DIR:-${repository_root}/build-unified}" \
  --output-root "${MESHAWARE_OUTPUT_ROOT:-${repository_root}/datasets}" \
  "$@"
