from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_data.artifacts import write_json_atomic
from meshaware_data.identifiers import matrix_id, sample_id
from meshaware_data.reporting import load_json_records, write_trial_report
from meshaware_data.schema import (
    AMG_SMOOTHERS,
    BOOMERAMG_PROFILES,
    PATTERNS,
)
from meshaware_data.solver import base_solver_command, solver_environment

SMOOTHER_LABELS = {
    "chebyshev": "Chebyshev",
    "damped-jacobi": "damped Jacobi",
    "l1-symmetric-gauss-seidel": r"$\ell_1$-sym. GS",
    "symmetric-gauss-seidel": "sym. GS",
}

PATTERN_LABELS = {
    "vertical_split": "vertical split",
    "checkerboard_2x2": r"checkerboard $2\times2$",
    "vertical_stripes_4": "four vertical stripes",
    "checkerboard_4x4": r"checkerboard $4\times4$",
}


@dataclass(frozen=True)
class AmgProfile:
    key: str
    label: str
    boomeramg_profile: str
    smoothers: tuple[str, ...]


@dataclass(frozen=True)
class SmootherStudyConfig:
    name: str
    mesh_family: str
    level: int
    patterns: tuple[str, ...]
    epsilons: tuple[float, ...]
    theta: float
    high_region: str
    relative_tolerance: float
    absolute_tolerance: float
    maximum_iterations: int
    jacobi_damping: float
    warmup_runs: int
    repeats: int
    table_smoothers: tuple[str, ...]
    amg_profiles: tuple[AmgProfile, ...]

    @property
    def h_nominal(self) -> float:
        return 2.0 ** (-self.level)

    @property
    def command_count(self) -> int:
        solver_variants = sum(len(profile.smoothers) for profile in self.amg_profiles)
        return len(self.patterns) * len(self.epsilons) * solver_variants

    @property
    def trial_count(self) -> int:
        return self.command_count * self.repeats

    def validate(self) -> None:
        if not self.name:
            raise ValueError("study name cannot be empty")
        if self.mesh_family != "polygonal":
            raise ValueError("the smoother study requires mesh_family='polygonal'")
        if self.level < 0 or self.level > 10:
            raise ValueError("level must lie in [0,10]")
        if not self.patterns or any(pattern not in PATTERNS for pattern in self.patterns):
            raise ValueError(f"patterns must be chosen from {PATTERNS}")
        if len(set(self.patterns)) != len(self.patterns):
            raise ValueError("patterns must be unique")
        if not self.epsilons or any(epsilon < 0.0 for epsilon in self.epsilons):
            raise ValueError("epsilons must be non-negative")
        if len(set(self.epsilons)) != len(self.epsilons):
            raise ValueError("epsilons must be unique")
        if not 0.0 < self.theta < 1.0:
            raise ValueError("theta must lie strictly in (0,1)")
        if self.high_region not in ("white", "gray"):
            raise ValueError("high_region must be 'white' or 'gray'")
        if self.relative_tolerance <= 0.0 or self.absolute_tolerance < 0.0:
            raise ValueError("solver tolerances must be valid")
        if self.maximum_iterations < 1 or self.repeats < 1:
            raise ValueError("maximum_iterations and repeats must be positive")
        if not 0.0 < self.jacobi_damping <= 1.0:
            raise ValueError("jacobi_damping must lie in (0,1]")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if not self.table_smoothers or any(
            smoother not in AMG_SMOOTHERS for smoother in self.table_smoothers
        ):
            raise ValueError(f"table_smoothers must be chosen from {AMG_SMOOTHERS}")
        if len(set(self.table_smoothers)) != len(self.table_smoothers):
            raise ValueError("table_smoothers must be unique")
        if not self.amg_profiles:
            raise ValueError("amg_profiles must not be empty")
        keys = [profile.key for profile in self.amg_profiles]
        if any(not key or "/" in key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("AMG profile keys must be unique path-safe strings")
        boomer_profiles = [profile.boomeramg_profile for profile in self.amg_profiles]
        if len(set(boomer_profiles)) != len(boomer_profiles):
            raise ValueError("boomeramg_profile values must be unique in this study")
        for profile in self.amg_profiles:
            if profile.boomeramg_profile not in BOOMERAMG_PROFILES:
                raise ValueError(
                    f"boomeramg_profile must be chosen from {BOOMERAMG_PROFILES}"
                )
            if not profile.smoothers or any(
                smoother not in self.table_smoothers for smoother in profile.smoothers
            ):
                raise ValueError(
                    f"profile {profile.key!r} smoothers must be present in table_smoothers"
                )
            if len(set(profile.smoothers)) != len(profile.smoothers):
                raise ValueError(f"profile {profile.key!r} smoothers must be unique")


@dataclass(frozen=True)
class StudyCase:
    profile: AmgProfile
    smoother: str
    pattern: str
    epsilon: float

    @property
    def label(self) -> str:
        return (
            f"{self.profile.key}/{self.smoother}/"
            f"{self.pattern}/epsilon={self.epsilon:g}"
        )


@dataclass(frozen=True)
class TableCell:
    status: str
    rho: float | None
    iterations: float | None
    setup_seconds: float | None
    solve_seconds: float | None
    repeats: int
    detail: str = ""


def summarize_failure(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in lines:
        if "HYPRE error code" in line:
            return line.split("PETSC ERROR:", maxsplit=1)[-1].strip()
    for line in reversed(lines):
        if line.lower().startswith("error:"):
            return line
    return lines[-1] if lines else "solver exited without a diagnostic"


def load_study_config(path: str | Path) -> SmootherStudyConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported smoother-study schema in {path}")
    profiles = tuple(
        AmgProfile(
            key=str(value["key"]),
            label=str(value["label"]),
            boomeramg_profile=str(value["boomeramg_profile"]),
            smoothers=tuple(str(item) for item in value["smoothers"]),
        )
        for value in raw["amg_profiles"]
    )
    config = SmootherStudyConfig(
        name=str(raw["name"]),
        mesh_family=str(raw["mesh_family"]),
        level=int(raw["level"]),
        patterns=tuple(str(value) for value in raw["patterns"]),
        epsilons=tuple(float(value) for value in raw["epsilons"]),
        theta=float(raw["theta"]),
        high_region=str(raw["high_region"]),
        relative_tolerance=float(raw["relative_tolerance"]),
        absolute_tolerance=float(raw["absolute_tolerance"]),
        maximum_iterations=int(raw["maximum_iterations"]),
        jacobi_damping=float(raw["jacobi_damping"]),
        warmup_runs=int(raw["warmup_runs"]),
        repeats=int(raw["repeats"]),
        table_smoothers=tuple(str(value) for value in raw["table_smoothers"]),
        amg_profiles=profiles,
    )
    config.validate()
    return config


def iter_cases(config: SmootherStudyConfig) -> Iterable[StudyCase]:
    for profile in config.amg_profiles:
        for smoother in profile.smoothers:
            for pattern in config.patterns:
                for epsilon in config.epsilons:
                    yield StudyCase(profile, smoother, pattern, epsilon)


def record_directory(data_root: Path, case: StudyCase) -> Path:
    return data_root / "records" / case.profile.key / case.smoother


def expected_record_paths(
    config: SmootherStudyConfig, data_root: Path, case: StudyCase
) -> tuple[Path, ...]:
    current_matrix_id = matrix_id(
        mesh_family=config.mesh_family,
        level=config.level,
        pattern=case.pattern,
        epsilon=case.epsilon,
        high_region=config.high_region,
    )
    directory = record_directory(data_root, case)
    return tuple(
        directory
        / f"{sample_id(current_matrix_id, theta=config.theta, repeat=repeat)}.json"
        for repeat in range(config.repeats)
    )


def failure_path(
    config: SmootherStudyConfig, data_root: Path, case: StudyCase
) -> Path:
    current_matrix_id = matrix_id(
        mesh_family=config.mesh_family,
        level=config.level,
        pattern=case.pattern,
        epsilon=case.epsilon,
        high_region=config.high_region,
    )
    return (
        data_root
        / "failures"
        / case.profile.key
        / case.smoother
        / f"{current_matrix_id}.json"
    )


def command_for_case(
    executable: Path,
    config: SmootherStudyConfig,
    data_root: Path,
    case: StudyCase,
    *,
    skip_existing_records: bool,
) -> list[str]:
    command = [
        *base_solver_command(
            executable,
            mesh_family=config.mesh_family,
            level=config.level,
            epsilon=case.epsilon,
            pattern=case.pattern,
            high_region=config.high_region,
            relative_tolerance=config.relative_tolerance,
            absolute_tolerance=config.absolute_tolerance,
            maximum_iterations=config.maximum_iterations,
        ),
        "--theta-values",
        str(config.theta),
        "--amg-backend",
        "boomeramg",
        "--boomeramg-profile",
        case.profile.boomeramg_profile,
        "--amg-smoother",
        case.smoother,
        "--jacobi-damping",
        str(config.jacobi_damping),
        "--repeats",
        str(config.repeats),
        "--warmup-runs",
        str(config.warmup_runs),
        "--record-dir",
        str(record_directory(data_root, case)),
        "--skip-matrix-write",
    ]
    if skip_existing_records:
        command.append("--skip-existing-records")
    return command


def _execute_case(
    index: int,
    total: int,
    case: StudyCase,
    command: list[str],
) -> tuple[int, StudyCase, list[str], int, str]:
    completed = subprocess.run(
        command,
        check=False,
        env=solver_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return index, case, command, completed.returncode, completed.stdout.strip()


def run_study(
    executable: Path,
    config: SmootherStudyConfig,
    data_root: Path,
    *,
    overwrite_records: bool,
    jobs: int,
) -> tuple[int, int, int]:
    cases = list(iter_cases(config))
    pending: list[tuple[int, StudyCase, list[str]]] = []
    skipped = 0
    for index, case in enumerate(cases, start=1):
        paths = expected_record_paths(config, data_root, case)
        failed_path = failure_path(config, data_root, case)
        if (
            all(path.is_file() for path in paths) or failed_path.is_file()
        ) and not overwrite_records:
            skipped += len(paths)
            continue
        if overwrite_records:
            for path in paths:
                path.unlink(missing_ok=True)
            failed_path.unlink(missing_ok=True)
        record_directory(data_root, case).mkdir(parents=True, exist_ok=True)
        pending.append(
            (
                index,
                case,
                command_for_case(
                    executable,
                    config,
                    data_root,
                    case,
                    skip_existing_records=not overwrite_records,
                ),
            )
        )

    if not pending:
        return 0, skipped, 0

    completed_trials = 0
    failed_cases = 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures: dict[
            Future[tuple[int, StudyCase, list[str], int, str]], StudyCase
        ] = {
            executor.submit(_execute_case, index, len(cases), case, command): case
            for index, case, command in pending
        }
        try:
            for future in as_completed(futures):
                index, case, command, returncode, output = future.result()
                if returncode != 0:
                    failed_cases += 1
                    destination = failure_path(config, data_root, case)
                    write_json_atomic(
                        destination,
                        {
                            "schema_version": 1,
                            "status": "solver_failed",
                            "profile_key": case.profile.key,
                            "boomeramg_profile": case.profile.boomeramg_profile,
                            "amg_smoother": case.smoother,
                            "pattern": case.pattern,
                            "epsilon": case.epsilon,
                            "level": config.level,
                            "h_nominal": config.h_nominal,
                            "theta": config.theta,
                            "returncode": returncode,
                            "summary": summarize_failure(output),
                            "command": command,
                            "output": output,
                        },
                    )
                    last_line = output.splitlines()[-1] if output else "no output"
                    print(
                        f"[{index}/{len(cases)}] {case.label}: FAILED: {last_line}",
                        flush=True,
                    )
                    continue
                for path in expected_record_paths(config, data_root, case):
                    if not path.is_file():
                        raise RuntimeError(
                            f"{case.label} did not produce expected record {path}"
                        )
                completed_trials += config.repeats
                last_line = output.splitlines()[-1] if output else "completed"
                print(f"[{index}/{len(cases)}] {case.label}: {last_line}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return completed_trials, skipped, failed_cases


def write_combined_report(
    config: SmootherStudyConfig, data_root: Path
) -> tuple[int, int]:
    paths = data_root.glob("records/*/*/*.json")
    records = load_json_records(paths)
    failures = list(data_root.glob("failures/*/*/*.json"))
    invalid = []
    for case in iter_cases(config):
        has_records = all(
            path.is_file() for path in expected_record_paths(config, data_root, case)
        )
        has_failure = failure_path(config, data_root, case).is_file()
        if has_records == has_failure:
            invalid.append(case.label)
    if invalid:
        raise RuntimeError(
            "every study case must have either records or one failure artifact; "
            f"invalid cases: {', '.join(invalid[:5])}"
        )
    write_trial_report(
        records,
        data_root / "diffusion_reports" / "trials.csv",
        config.name,
    )
    return len(records), len(failures)


def _profile_by_boomer_name(
    config: SmootherStudyConfig,
) -> dict[str, AmgProfile]:
    return {profile.boomeramg_profile: profile for profile in config.amg_profiles}


def aggregate_cells(
    config: SmootherStudyConfig, data_root: Path
) -> dict[tuple[str, float, str, str], TableCell]:
    profile_by_name = _profile_by_boomer_name(config)
    grouped: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in load_json_records(data_root.glob("records/*/*/*.json")):
        profile_name = str(record.get("boomeramg_profile", "default"))
        try:
            profile = profile_by_name[profile_name]
        except KeyError as error:
            raise ValueError(f"record has unexpected AMG profile {profile_name!r}") from error
        key = (
            str(record["pattern"]),
            float(record["epsilon"]),
            profile.key,
            str(record["amg_smoother"]),
        )
        grouped[key].append(record)

    cells: dict[tuple[str, float, str, str], TableCell] = {}
    for key, records in grouped.items():
        cells[key] = TableCell(
            status="ok",
            rho=statistics.mean(float(record["convergence_factor"]) for record in records),
            iterations=statistics.mean(float(record["cg_iterations"]) for record in records),
            setup_seconds=statistics.mean(
                float(record["amg_setup_time_seconds"]) for record in records
            ),
            solve_seconds=statistics.mean(
                float(record["solve_time_seconds"]) for record in records
            ),
            repeats=len(records),
        )
    for path in sorted(data_root.glob("failures/*/*/*.json")):
        with path.open(encoding="utf-8") as handle:
            failure = json.load(handle)
        key = (
            str(failure["pattern"]),
            float(failure["epsilon"]),
            str(failure["profile_key"]),
            str(failure["amg_smoother"]),
        )
        if key in cells:
            raise ValueError(f"study cell has both records and a failure artifact: {key}")
        cells[key] = TableCell(
            status="solver_failed",
            rho=None,
            iterations=None,
            setup_seconds=None,
            solve_seconds=None,
            repeats=0,
            detail=str(
                failure.get(
                    "summary", summarize_failure(str(failure.get("output", "")))
                )
            ),
        )
    return cells


def _cell_color(rho: float, rho_min: float, rho_max: float) -> str:
    t = 0.0 if rho_max <= rho_min else (rho - rho_min) / (rho_max - rho_min)
    low = (0.42, 0.60, 0.88)
    middle = (0.97, 0.97, 0.97)
    high = (0.94, 0.50, 0.42)
    if t <= 0.5:
        start, stop, local = low, middle, 2.0 * t
    else:
        start, stop, local = middle, high, 2.0 * (t - 0.5)
    return ",".join(
        f"{start[index] + (stop[index] - start[index]) * local:.3f}"
        for index in range(3)
    )


def _format_cell(
    cell: TableCell | None,
    *,
    compatible: bool,
    best_rho: float | None,
    rho_min: float,
    rho_max: float,
) -> str:
    if not compatible:
        return "n/a"
    if cell is None:
        return "--"
    if cell.status != "ok":
        return r"failed\textsuperscript{\ensuremath{\ddagger}}"
    assert cell.rho is not None and cell.iterations is not None
    value = f"{cell.rho:.3f} ({int(round(cell.iterations))})"
    if best_rho is not None and math.isclose(cell.rho, best_rho, abs_tol=5.0e-13):
        value = rf"\textbf{{{value}}}"
    return rf"\cellcolor[rgb]{{{_cell_color(cell.rho, rho_min, rho_max)}}}{value}"


def _table_block(
    config: SmootherStudyConfig,
    pattern: str,
    cells: dict[tuple[str, float, str, str], TableCell],
    rho_min: float,
    rho_max: float,
) -> str:
    profile_headers = " & ".join(
        rf"\multicolumn{{{len(config.table_smoothers)}}}{{c}}{{{profile.label}}}"
        for profile in config.amg_profiles
    )
    cmidrules: list[str] = []
    first_column = 2
    for _profile in config.amg_profiles:
        last_column = first_column + len(config.table_smoothers) - 1
        cmidrules.append(rf"\cmidrule(lr){{{first_column}-{last_column}}}")
        first_column = last_column + 1
    smoother_headers = " & ".join(
        SMOOTHER_LABELS[smoother]
        for _profile in config.amg_profiles
        for smoother in config.table_smoothers
    )
    rows: list[str] = []
    for epsilon in config.epsilons:
        row_cells = [
            cells.get((pattern, epsilon, profile.key, smoother))
            for profile in config.amg_profiles
            for smoother in config.table_smoothers
            if smoother in profile.smoothers
        ]
        best_rho = min(
            (
                cell.rho
                for cell in row_cells
                if cell is not None and cell.rho is not None
            ),
            default=None,
        )
        formatted = [
            _format_cell(
                cells.get((pattern, epsilon, profile.key, smoother)),
                compatible=smoother in profile.smoothers,
                best_rho=best_rho,
                rho_min=rho_min,
                rho_max=rho_max,
            )
            for profile in config.amg_profiles
            for smoother in config.table_smoothers
        ]
        rows.append(f"{epsilon:g} & " + " & ".join(formatted) + r" \\")

    columns = "c" + "c" * (
        len(config.amg_profiles) * len(config.table_smoothers)
    )
    pattern_label = PATTERN_LABELS.get(pattern, pattern.replace("_", " "))
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\renewcommand{\arraystretch}{1.10}",
            r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            rf" & {profile_headers} \\",
            "".join(cmidrules),
            rf"$\varepsilon$ & {smoother_headers} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}}",
            (
                r"\caption{Polygonal SIPG, "
                + pattern_label
                + rf", refinement $L={config.level}$ ($h=2^{{-{config.level}}}$), "
                + rf"and fixed $\theta={config.theta:g}$. Each entry is mean $\rho$ "
                + r"with mean CG iterations in parentheses; the best successful $\rho$ in each row is bold.}"
            ),
            rf"\label{{tab:polygonal-smoothers-{pattern.replace('_', '-')}}}",
            r"\end{table}",
        ]
    )


def latex_document(
    config: SmootherStudyConfig,
    cells: dict[tuple[str, float, str, str], TableCell],
) -> str:
    expected_keys = {
        (pattern, epsilon, profile.key, smoother)
        for pattern in config.patterns
        for epsilon in config.epsilons
        for profile in config.amg_profiles
        for smoother in profile.smoothers
    }
    missing = sorted(expected_keys.difference(cells))
    if missing:
        preview = ", ".join(str(key) for key in missing[:5])
        raise ValueError(
            f"cannot build complete smoother tables: {len(missing)} cells missing; {preview}"
        )
    rho_values = [
        cells[key].rho for key in expected_keys if cells[key].rho is not None
    ]
    if not rho_values:
        raise ValueError("the study contains no successful solver results")
    rho_min, rho_max = min(rho_values), max(rho_values)
    tables = [
        _table_block(config, pattern, cells, rho_min, rho_max)
        for pattern in config.patterns
    ]
    failure_note = []
    if any(cells[key].status != "ok" for key in expected_keys):
        failure_note = [
            (
                r"\par\noindent\textsuperscript{\ensuremath{\ddagger}}The solver returned a "
                r"nonzero status; the exact command and diagnostic are retained in "
                r"the corresponding JSON failure artifact."
            )
        ]
    return "\n\n".join(
        [
            r"\documentclass[a4paper]{article}",
            r"\usepackage[margin=0.55in]{geometry}",
            r"\usepackage[table]{xcolor}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
            r"\usepackage{float}",
            r"\begin{document}",
            r"\section*{Polygonal smoother and AMG-representation study}",
            (
                r"All tables use the same assembled modal \texttt{FE\_AggloDGP(1)} "
                r"operator. ``Scalar AMG'' is the default BoomerAMG treatment of "
                r"its modal coefficients. ``Nodal AMG profile'' keeps that modal "
                r"operator but coarsens each polygon's modal block nodally. "
                rf"Damped Jacobi uses $\omega={config.jacobi_damping:g}$. "
                r"Blue-to-red shading uses one $\rho$ range "
                r"across every table."
            ),
            *tables,
            *failure_note,
            r"\end{document}",
        ]
    ) + "\n"


def write_summary_csv(
    config: SmootherStudyConfig,
    cells: dict[tuple[str, float, str, str], TableCell],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pattern",
                "epsilon",
                "level",
                "h_nominal",
                "theta",
                "profile_key",
                "profile_label",
                "boomeramg_profile",
                "amg_smoother",
                "status",
                "rho_mean",
                "iterations_mean",
                "amg_setup_seconds_mean",
                "solve_seconds_mean",
                "repeats",
                "detail",
            ),
        )
        writer.writeheader()
        for pattern in config.patterns:
            for epsilon in config.epsilons:
                for profile in config.amg_profiles:
                    for smoother in profile.smoothers:
                        cell = cells[(pattern, epsilon, profile.key, smoother)]
                        writer.writerow(
                            {
                                "pattern": pattern,
                                "epsilon": epsilon,
                                "level": config.level,
                                "h_nominal": config.h_nominal,
                                "theta": config.theta,
                                "profile_key": profile.key,
                                "profile_label": profile.label,
                                "boomeramg_profile": profile.boomeramg_profile,
                                "amg_smoother": smoother,
                                "status": cell.status,
                                "rho_mean": "" if cell.rho is None else cell.rho,
                                "iterations_mean": (
                                    "" if cell.iterations is None else cell.iterations
                                ),
                                "amg_setup_seconds_mean": (
                                    ""
                                    if cell.setup_seconds is None
                                    else cell.setup_seconds
                                ),
                                "solve_seconds_mean": (
                                    "" if cell.solve_seconds is None else cell.solve_seconds
                                ),
                                "repeats": cell.repeats,
                                "detail": cell.detail,
                            }
                        )


def generate_outputs(
    config: SmootherStudyConfig, data_root: Path, figure_root: Path, *, compile_pdf: bool
) -> tuple[Path, Path]:
    cells = aggregate_cells(config, data_root)
    figure_root.mkdir(parents=True, exist_ok=True)
    csv_path = figure_root / "polygonal_smoother_summary.csv"
    tex_path = figure_root / "polygonal_smoother_tables.tex"
    write_summary_csv(config, cells, csv_path)
    tex_path.write_text(latex_document(config, cells), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {tex_path}")
    if compile_pdf:
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            raise RuntimeError("--compile requested but pdflatex is not installed")
        for _ in range(2):
            completed = subprocess.run(
                [
                    pdflatex,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={figure_root}",
                    str(tex_path),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"pdflatex failed for {tex_path}:\n{completed.stdout}"
                )
        print(f"wrote {tex_path.with_suffix('.pdf')}")
    return csv_path, tex_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the level-5 polygonal BoomerAMG smoother/profile study and "
            "generate publication-ready LaTeX tables."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "polygonal_smoother_level5.json",
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build-unified")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument(
        "--figure-root", type=Path, default=REPO_ROOT / "results" / "figures"
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite-records", action="store_true")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Regenerate CSV/TeX from a complete existing record tree.",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    config = load_study_config(args.config)
    data_root = args.output_root / config.name
    figure_root = args.figure_root / config.name
    executable = args.build_dir / "meshaware_diffusion_polydeal"

    if args.dry_run:
        cases = list(iter_cases(config))
        print(
            json.dumps(
                {
                    "config": str(args.config),
                    "data_root": str(data_root),
                    "figure_root": str(figure_root),
                    "level": config.level,
                    "h_nominal": config.h_nominal,
                    "theta": config.theta,
                    "epsilons": config.epsilons,
                    "patterns": config.patterns,
                    "command_count": config.command_count,
                    "trial_count": config.trial_count,
                    "profiles": [asdict(profile) for profile in config.amg_profiles],
                    "first_command": command_for_case(
                        executable,
                        config,
                        data_root,
                        cases[0],
                        skip_existing_records=True,
                    ),
                },
                indent=2,
            )
        )
        return

    if not args.generate_only:
        if not executable.is_file():
            raise SystemExit(
                f"Missing executable {executable}. Build the PolyDeal driver first."
            )
        data_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "source_config": str(args.config.resolve()),
            "expanded_config": asdict(config),
            "command_count": config.command_count,
            "trial_count": config.trial_count,
            "record_layout": "records/<profile-key>/<smoother>/<sample-id>.json",
        }
        write_json_atomic(data_root / "manifest.json", manifest)
        completed, skipped, failed = run_study(
            executable,
            config,
            data_root,
            overwrite_records=args.overwrite_records,
            jobs=args.jobs,
        )
        reported, recorded_failures = write_combined_report(config, data_root)
        print(
            f"study complete: completed={completed}, skipped={skipped}, "
            f"new_failures={failed}, reported={reported}, "
            f"total_failures={recorded_failures}"
        )

    generate_outputs(config, data_root, figure_root, compile_pdf=args.compile)


if __name__ == "__main__":
    main()
