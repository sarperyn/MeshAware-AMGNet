#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_figure_pipeline.sh [options]

Options:
  --dataset-root DIR   Experiment root containing FAMILY/diffusion_reports.
                       Default: datasets/small
  --output-root DIR    Root directory for generated figures.
                       Default: results/figures/small
  PYTHON=PATH          Python interpreter with requirements-figures installed.
                       Default: python3
  --no-pdf             Skip pdflatex compilation.
  --help               Show this message.

The script generates separate outputs for every mesh family, including the
theta-vs-hierarchy-level diagnostic, and compiles generated .tex files to PDF
when possible.
USAGE
}

dataset_root="datasets/medium"
output_root="results/figures/medium"
compile_pdf=1
python_bin="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      dataset_root="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --no-pdf)
      compile_pdf=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p "$output_root"
failures=0
compile_failures=0

echo "Dataset root: $dataset_root"
echo "Output root:  $output_root"
echo

run_step() {
  local name="$1"
  shift
  echo "==> $name"
  if "$@"; then
    :
  else
    echo "WARNING: step failed: $name" >&2
    failures=$((failures + 1))
  fi
  echo
}

compile_tex_tree() {
  local root="$1"
  if ! command -v pdflatex >/dev/null 2>&1; then
    echo "WARNING: pdflatex not found; skipping TeX to PDF compilation." >&2
    return 0
  fi

  local tex_file
  while IFS= read -r tex_file; do
    local tex_dir
    local tex_name
    tex_dir="$(dirname "$tex_file")"
    tex_name="$(basename "$tex_file")"
    if grep -q '\\usepackage{pgfplots}' "$tex_file" &&
       ! kpsewhich pgfplots.sty >/dev/null 2>&1; then
      echo "Skipping $tex_file: pgfplots.sty is not installed (SVG exists)."
      echo
      continue
    fi
    echo "==> Compiling $tex_file"
    if (
      cd "$tex_dir"
      pdflatex -interaction=nonstopmode -halt-on-error "$tex_name" >/dev/null
      pdflatex -interaction=nonstopmode -halt-on-error "$tex_name" >/dev/null
    ); then
      echo "Wrote ${tex_file%.tex}.pdf"
    else
      echo "WARNING: Failed to compile $tex_file. Check ${tex_file%.tex}.log." >&2
      compile_failures=$((compile_failures + 1))
    fi
    echo
  done < <(find "$root" -name '*.tex' -type f | sort)
}

family_count=0
for report_path in "${dataset_root%/}"/*/diffusion_reports/trials.csv; do
  [[ -f "$report_path" ]] || continue
  family="$(basename "$(dirname "$(dirname "$report_path")")")"
  family_root="$output_root/$family"
  family_count=$((family_count + 1))

  echo "---- Mesh family: $family ----"
  run_step "$family theta-rho tables" \
    "$python_bin" scripts/generate_theta_rho_tables.py \
      --input_glob "$report_path" \
      --out_dir "$family_root/theta_rho_relation"

  run_step "$family theta-cost plots" \
    "$python_bin" scripts/generate_theta_cost_plots.py \
      --input-glob "$report_path" \
      --out-dir "$family_root/theta_cost_relation" \
      --svg

  run_step "$family rho-time scatter" \
    "$python_bin" scripts/generate_rho_time_scatter.py \
      --input-glob "$report_path" \
      --out-dir "$family_root/rho_time_scatter" \
      --csv-name "rho_time_scatter_small.csv" \
      --svg-name "rho_time_scatter_small.svg" \
      --no-png

  run_step "$family theta vs hierarchy levels" \
    "$python_bin" scripts/generate_theta_vs_nlevels_plot.py \
      --dataset-root "${dataset_root%/}/$family" \
      --output-dir "$family_root/theta_nlevels" \
      --include-splits all
done

if [[ "$family_count" -eq 0 ]]; then
  echo "No family diffusion reports found under $dataset_root" >&2
  exit 1
fi

if [[ "$compile_pdf" -eq 1 ]]; then
  compile_tex_tree "$output_root"
fi

echo "Done. Outputs are under $output_root"
if [[ "$compile_failures" -gt 0 ]]; then
  echo "PDF compilation completed with $compile_failures warning(s)." >&2
fi
if [[ "$failures" -gt 0 ]]; then
  echo "Completed with $failures failure(s)." >&2
  exit 1
fi
