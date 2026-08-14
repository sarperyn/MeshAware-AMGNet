#include "meshaware/coefficient_patterns.hpp"
#include "meshaware/driver_cli.hpp"
#include "meshaware/experiment_record.hpp"
#include "meshaware/petsc_amg.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

enum class MeshFamily { quadrilateral, simplex };

MeshFamily parse_mesh_family(const std::string_view name) {
  if (name == "quadrilateral")
    return MeshFamily::quadrilateral;
  if (name == "simplex")
    return MeshFamily::simplex;
  throw std::invalid_argument(
      "deal.II mesh family must be 'quadrilateral' or 'simplex'");
}

const char *to_string(const MeshFamily family) {
  return family == MeshFamily::quadrilateral ? "quadrilateral" : "simplex";
}

const char *matrix_prefix(const MeshFamily family) {
  return family == MeshFamily::quadrilateral ? "quad" : "simplex";
}

double seconds_since(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Options {
  MeshFamily mesh_family = MeshFamily::quadrilateral;
  unsigned int level = 3;
  double epsilon = 0.0;
  meshaware::Pattern pattern = meshaware::Pattern::vertical_split;
  meshaware::HighRegion high_region = meshaware::HighRegion::white;
  double theta = 0.24;
  double relative_tolerance = 1e-8;
  double absolute_tolerance = 1e-50;
  unsigned int maximum_iterations = 10000;
  meshaware::AmgSmoother amg_smoother =
      meshaware::AmgSmoother::symmetric_gauss_seidel;
  double jacobi_damping = 2.0 / 3.0;
  unsigned int repeat = 0;
  std::vector<double> theta_values;
  unsigned int repeats = 1;
  unsigned int warmup_runs = 0;
  std::filesystem::path record_path;
  std::filesystem::path record_directory;
  std::filesystem::path matrix_path;
  bool skip_matrix_write = false;
  bool skip_existing_records = false;
  bool assemble_only = false;
};

Options parse_options(const int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--mesh-family")
      options.mesh_family =
          parse_mesh_family(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--level")
      options.level =
          std::stoul(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--epsilon")
      options.epsilon =
          std::stod(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--pattern")
      options.pattern = meshaware::parse_pattern(
          meshaware::require_option_value(i, argc, argv));
    else if (argument == "--high-region")
      options.high_region =
          meshaware::parse_high_region(
              meshaware::require_option_value(i, argc, argv));
    else if (argument == "--theta")
      options.theta =
          std::stod(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--theta-values")
      options.theta_values = meshaware::parse_float_list(
          meshaware::require_option_value(i, argc, argv), "theta-values");
    else if (argument == "--rtol")
      options.relative_tolerance =
          std::stod(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--atol")
      options.absolute_tolerance =
          std::stod(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--max-iterations")
      options.maximum_iterations =
          std::stoul(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--amg-smoother")
      options.amg_smoother =
          meshaware::parse_amg_smoother(
              meshaware::require_option_value(i, argc, argv));
    else if (argument == "--jacobi-damping")
      options.jacobi_damping =
          std::stod(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--repeat")
      options.repeat =
          std::stoul(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--repeats")
      options.repeats =
          std::stoul(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--warmup-runs")
      options.warmup_runs =
          std::stoul(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--record")
      options.record_path = meshaware::require_option_value(i, argc, argv);
    else if (argument == "--record-dir")
      options.record_directory =
          meshaware::require_option_value(i, argc, argv);
    else if (argument == "--matrix")
      options.matrix_path = meshaware::require_option_value(i, argc, argv);
    else if (argument == "--skip-matrix-write")
      options.skip_matrix_write = true;
    else if (argument == "--skip-existing-records")
      options.skip_existing_records = true;
    else if (argument == "--assemble-only")
      options.assemble_only = true;
    else if (argument == "--help") {
      std::cout
          << "Usage: meshaware_diffusion_dealii [options]\n"
          << "  --mesh-family NAME     quadrilateral or simplex\n"
          << "  --level N              nominal h = 2^-N\n"
          << "  --epsilon E            coefficient contrast = 10^E\n"
          << "  --pattern NAME         vertical_split, checkerboard_2x2,\n"
          << "                         vertical_stripes_4, checkerboard_4x4\n"
          << "  --high-region NAME     white (Eq. 15) or gray (Fig. 1 "
             "caption)\n"
          << "  --theta T              BoomerAMG strong threshold\n"
          << "  --theta-values CSV     batched strong-threshold grid\n"
          << "  --rtol T --atol T --max-iterations N\n"
          << "  --amg-smoother NAME    chebyshev, damped-jacobi, "
             "l1-symmetric-gauss-seidel, or symmetric-gauss-seidel\n"
          << "  --jacobi-damping W     damping in (0,1], default 2/3\n"
          << "  --repeat N              timing repeat identifier\n"
          << "  --repeats N             repeats per theta in batch mode\n"
          << "  --warmup-runs N          discarded warm-ups per theta\n"
          << "  --record PATH           JSON trial record\n"
          << "  --record-dir PATH       generated records for batch mode\n"
          << "  --matrix PATH           PETSc binary matrix/reference\n"
          << "  --skip-matrix-write     record path without rewriting matrix\n"
          << "  --skip-existing-records resume a partial batch\n"
          << "  --assemble-only         export matrix without solving\n";
      std::exit(0);
    } else
      throw std::invalid_argument("Unknown argument: " + argument);
  }

  if (options.theta_values.empty())
    options.theta_values.push_back(options.theta);
  if (std::any_of(
          options.theta_values.begin(), options.theta_values.end(),
          [](const double theta) { return !(theta > 0.0 && theta < 1.0); }))
    throw std::invalid_argument("all theta values must lie strictly in (0,1)");
  if (options.epsilon < 0.0)
    throw std::invalid_argument("epsilon must be non-negative");
  if (options.level > 20)
    throw std::invalid_argument("level is unreasonably large");
  if (options.relative_tolerance <= 0.0 || options.absolute_tolerance < 0.0 ||
      options.maximum_iterations == 0)
    throw std::invalid_argument("invalid solver tolerances or iteration limit");
  if (!(options.jacobi_damping > 0.0 && options.jacobi_damping <= 1.0))
    throw std::invalid_argument("jacobi-damping must lie in (0,1]");
  if (options.repeats == 0)
    throw std::invalid_argument("repeats must be positive");
  if (!options.record_directory.empty() && !options.record_path.empty())
    throw std::invalid_argument("record and record-dir are mutually exclusive");
  if (options.record_directory.empty() && options.theta_values.size() != 1)
    throw std::invalid_argument("theta-values requires record-dir batch mode");
  if (options.assemble_only &&
      (options.matrix_path.empty() || options.skip_matrix_write))
    throw std::invalid_argument(
        "assemble-only requires a writable --matrix path");
  if (options.assemble_only &&
      (!options.record_path.empty() || !options.record_directory.empty()))
    throw std::invalid_argument(
        "assemble-only does not accept record output options");
  return options;
}

class ExactSolution : public dealii::Function<2> {
public:
  explicit ExactSolution(const meshaware::Pattern pattern)
      : dealii::Function<2>(1), pattern(pattern) {}

  double value(const dealii::Point<2> &point,
               const unsigned int component = 0) const override {
    (void)component;
    return meshaware::exact_value(pattern, point[0], point[1]);
  }

private:
  const meshaware::Pattern pattern;
};

struct ErrorNorms {
  double l2;
  double h1_seminorm;
  double energy;
};

class DiffusionExperiment {
public:
  explicit DiffusionExperiment(const Options &options)
      : options(options), finite_element(make_finite_element()),
        dof_handler(triangulation) {}

  void run() {
    make_grid();
    setup_system();

    const auto assembly_start = Clock::now();
    assemble_system();
    const double assembly_seconds = seconds_since(assembly_start);

    const std::uint64_t nonzeros = matrix_nonzeros();
    if (!options.matrix_path.empty() && !options.skip_matrix_write)
      meshaware::write_petsc_matrix(static_cast<Mat>(system_matrix),
                                    options.matrix_path);

    if (options.assemble_only) {
      std::cout << "assembled_only=1 dofs=" << dof_handler.n_dofs()
                << " nnz=" << nonzeros << '\n';
      return;
    }

    if (options.record_directory.empty())
      run_trial(options.theta_values.front(), options.repeat,
                options.record_path, assembly_seconds, nonzeros);
    else
      run_batch(assembly_seconds, nonzeros);
  }

private:
  void run_batch(const double assembly_seconds, const std::uint64_t nonzeros) {
    unsigned int completed = 0;
    unsigned int skipped = 0;
    for (const double theta : options.theta_values) {
      std::vector<unsigned int> pending_repeats;
      for (unsigned int repeat = 0; repeat < options.repeats; ++repeat) {
        const auto path = batch_record_path(theta, repeat);
        if (options.skip_existing_records && std::filesystem::exists(path)) {
          ++skipped;
        } else {
          pending_repeats.push_back(repeat);
        }
      }
      if (pending_repeats.empty())
        continue;

      for (unsigned int warmup = 0; warmup < options.warmup_runs; ++warmup)
        (void)solve(theta);

      for (const unsigned int repeat : pending_repeats) {
        run_trial(theta, repeat, batch_record_path(theta, repeat),
                  assembly_seconds, nonzeros);
        ++completed;
      }
    }
    std::cout << "batch_completed=" << completed << " batch_skipped=" << skipped
              << '\n';
  }

  void run_trial(const double theta, const unsigned int repeat,
                 const std::filesystem::path &record_path,
                 const double assembly_seconds, const std::uint64_t nonzeros) {
    const meshaware::SolverMetrics solver_metrics = solve(theta);
    const ErrorNorms errors = compute_errors();
    write_record(theta, repeat, record_path, assembly_seconds, solver_metrics,
                 errors, nonzeros);

    std::cout << std::setprecision(8)
              << "cells=" << triangulation.n_active_cells()
              << " dofs=" << dof_handler.n_dofs() << " h_max=" << h_max
              << " nnz=" << nonzeros << " theta=" << theta
              << " smoother=" << meshaware::to_string(options.amg_smoother)
              << " repeat=" << repeat
              << " iterations=" << solver_metrics.iterations
              << " amg_levels=" << solver_metrics.amg_levels
              << " rho=" << solver_metrics.convergence_factor
              << " setup_s=" << solver_metrics.setup_seconds
              << " solve_s=" << solver_metrics.solve_seconds
              << " grid_complexity=" << solver_metrics.grid_complexity
              << " operator_complexity="
              << solver_metrics.operator_complexity << '\n';
  }

  void make_grid() {
    const unsigned int subdivisions = 1u << (options.level + 1);
    if (options.mesh_family == MeshFamily::quadrilateral)
      dealii::GridGenerator::subdivided_hyper_rectangle(
          triangulation, std::vector<unsigned int>(2, subdivisions),
          dealii::Point<2>(-1.0, -1.0), dealii::Point<2>(1.0, 1.0), false);
    else {
      dealii::Triangulation<2> quadrilateral_grid;
      dealii::GridGenerator::subdivided_hyper_rectangle(
          quadrilateral_grid, std::vector<unsigned int>(2, subdivisions),
          dealii::Point<2>(-1.0, -1.0), dealii::Point<2>(1.0, 1.0), false);
      dealii::GridGenerator::convert_hypercube_to_simplex_mesh(
          quadrilateral_grid, triangulation, 2);
    }

    h_max = 0.0;
    for (const auto &cell : triangulation.active_cell_iterators())
      h_max = std::max(h_max, cell->diameter());
  }

  std::unique_ptr<dealii::FiniteElement<2>> make_finite_element() const {
    if (options.mesh_family == MeshFamily::quadrilateral)
      return std::make_unique<dealii::FE_Q<2>>(1);
    return std::make_unique<dealii::FE_SimplexP<2>>(1);
  }

  dealii::Quadrature<2>
  make_quadrature(const unsigned int points_per_direction) const {
    if (options.mesh_family == MeshFamily::quadrilateral)
      return dealii::QGauss<2>(points_per_direction);
    return dealii::QGaussSimplex<2>(points_per_direction);
  }

  void setup_system() {
    dof_handler.distribute_dofs(*finite_element);

    dealii::VectorTools::interpolate_boundary_values(
        dof_handler, 0, ExactSolution(options.pattern), constraints);
    constraints.close();

    dealii::DynamicSparsityPattern dynamic_pattern(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dynamic_pattern,
                                            constraints, false);
    sparsity_pattern.copy_from(dynamic_pattern);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(PETSC_COMM_SELF, dof_handler.n_dofs(),
                    dof_handler.n_dofs());
    right_hand_side.reinit(PETSC_COMM_SELF, dof_handler.n_dofs(),
                           dof_handler.n_dofs());
  }

  double coefficient(const dealii::Point<2> &point) const {
    return meshaware::diffusion_coefficient(options.pattern, options.epsilon,
                                            point[0], point[1],
                                            options.high_region);
  }

  void assemble_system() {
    const dealii::Quadrature<2> quadrature =
        make_quadrature(finite_element->degree + 1);
    dealii::FEValues<2> fe_values(
        *finite_element, quadrature,
        dealii::update_values | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = finite_element->n_dofs_per_cell();
    const unsigned int n_q_points = quadrature.size();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      fe_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const auto &point = fe_values.quadrature_point(q);
        const double mu = coefficient(point);
        const double rhs =
            meshaware::forcing_value(options.pattern, options.epsilon, point[0],
                                     point[1], options.high_region);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          cell_rhs(i) += fe_values.shape_value(i, q) * rhs * fe_values.JxW(q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += mu * fe_values.shape_grad(i, q) *
                                 fe_values.shape_grad(j, q) * fe_values.JxW(q);
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                             local_dof_indices, system_matrix,
                                             right_hand_side);
    }

    system_matrix.compress(dealii::VectorOperation::add);
    right_hand_side.compress(dealii::VectorOperation::add);
  }

  std::uint64_t matrix_nonzeros() const {
    MatInfo information;
    meshaware::petsc_check(MatGetInfo(static_cast<Mat>(system_matrix),
                                      MAT_GLOBAL_SUM, &information),
                           "MatGetInfo");
    return static_cast<std::uint64_t>(information.nz_used);
  }

  meshaware::SolverMetrics solve(const double theta) {
    const meshaware::AmgSolverOptions solver_options{
        options.relative_tolerance, options.absolute_tolerance,
        options.maximum_iterations, theta, options.amg_smoother,
        options.jacobi_damping};
    meshaware::SolverMetrics metrics = meshaware::solve_with_boomer_amg(
        static_cast<Mat>(system_matrix),
        static_cast<const Vec &>(right_hand_side),
        static_cast<const Vec &>(solution), solver_options);
    constraints.distribute(solution);
    return metrics;
  }

  ErrorNorms compute_errors() const {
    const dealii::Quadrature<2> quadrature =
        make_quadrature(finite_element->degree + 2);
    dealii::FEValues<2> fe_values(
        *finite_element, quadrature,
        dealii::update_values | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<double> values(quadrature.size());
    std::vector<dealii::Tensor<1, 2>> gradients(quadrature.size());
    double l2_squared = 0.0;
    double h1_squared = 0.0;
    double energy_squared = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      fe_values.reinit(cell);
      fe_values.get_function_values(solution, values);
      fe_values.get_function_gradients(solution, gradients);
      for (unsigned int q = 0; q < quadrature.size(); ++q) {
        const auto &point = fe_values.quadrature_point(q);
        const double value_error =
            values[q] -
            meshaware::exact_value(options.pattern, point[0], point[1]);
        const auto exact_gradient =
            meshaware::exact_gradient(options.pattern, point[0], point[1]);
        const double gradient_error_x = gradients[q][0] - exact_gradient[0];
        const double gradient_error_y = gradients[q][1] - exact_gradient[1];
        const double gradient_error_squared =
            gradient_error_x * gradient_error_x +
            gradient_error_y * gradient_error_y;

        l2_squared += value_error * value_error * fe_values.JxW(q);
        h1_squared += gradient_error_squared * fe_values.JxW(q);
        energy_squared +=
            coefficient(point) * gradient_error_squared * fe_values.JxW(q);
      }
    }

    return {std::sqrt(l2_squared), std::sqrt(h1_squared),
            std::sqrt(energy_squared)};
  }

  std::string matrix_id() const {
    return meshaware::make_matrix_id(
        matrix_prefix(options.mesh_family), options.level,
        meshaware::to_string(options.pattern), options.epsilon,
        meshaware::to_string(options.high_region));
  }

  std::filesystem::path batch_record_path(const double theta,
                                          const unsigned int repeat) const {
    const std::string sample_id =
        meshaware::make_sample_id(matrix_id(), theta, repeat);
    return options.record_directory / (sample_id + ".json");
  }

  void write_record(const double theta, const unsigned int repeat,
                    const std::filesystem::path &record_path,
                    const double assembly_seconds,
                    const meshaware::SolverMetrics &solver_metrics,
                    const ErrorNorms &errors,
                    const std::uint64_t nonzeros) const {
    const std::string id = matrix_id();
    meshaware::ExperimentRecord record;
    record.sample_id = meshaware::make_sample_id(id, theta, repeat);
    record.matrix_id = id;
    record.mesh_family = to_string(options.mesh_family);
    record.level = options.level;
    record.h_nominal = std::pow(2.0, -int(options.level));
    record.h_max = h_max;
    record.pattern = meshaware::to_string(options.pattern);
    record.epsilon = options.epsilon;
    record.high_region = meshaware::to_string(options.high_region);
    record.theta = theta;
    record.amg_smoother = meshaware::to_string(options.amg_smoother);
    record.amg_relaxation_weight =
        options.amg_smoother == meshaware::AmgSmoother::damped_jacobi
            ? options.jacobi_damping
            : 1.0;
    record.repeat = repeat;
    record.cells = triangulation.n_active_cells();
    record.background_cells = triangulation.n_active_cells();
    record.dofs = dof_handler.n_dofs();
    record.nonzeros = nonzeros;
    record.cg_iterations = solver_metrics.iterations;
    record.amg_levels = solver_metrics.amg_levels;
    record.ksp_converged_reason = static_cast<int>(solver_metrics.reason);
    record.residual_initial = solver_metrics.residual_initial;
    record.residual_final = solver_metrics.residual_final;
    record.convergence_factor = solver_metrics.convergence_factor;
    record.grid_complexity = solver_metrics.grid_complexity;
    record.operator_complexity = solver_metrics.operator_complexity;
    record.l2_error = errors.l2;
    record.h1_seminorm_error = errors.h1_seminorm;
    record.energy_error = errors.energy;
    record.assembly_time_seconds = assembly_seconds;
    record.amg_setup_time_seconds = solver_metrics.setup_seconds;
    record.solve_time_seconds = solver_metrics.solve_seconds;
    record.matrix_path = options.matrix_path;
    meshaware::write_experiment_record(record, record_path);
  }

  const Options options;
  dealii::Triangulation<2> triangulation;
  std::unique_ptr<dealii::FiniteElement<2>> finite_element;
  dealii::DoFHandler<2> dof_handler;
  dealii::AffineConstraints<double> constraints;
  dealii::SparsityPattern sparsity_pattern;
  dealii::PETScWrappers::SparseMatrix system_matrix;
  dealii::PETScWrappers::MPI::Vector solution;
  dealii::PETScWrappers::MPI::Vector right_hand_side;
  double h_max = 0.0;
};
} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);

    // PETSc parses argv during initialization. Pass only the program name
    // because all application options above are handled by this executable.
    // This avoids PETSc reporting valid application options as unused.
    int initialization_argc = 1;
    char *initialization_arguments[] = {argv[0], nullptr};
    char **initialization_argv = initialization_arguments;
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        initialization_argc, initialization_argv, 1);

    DiffusionExperiment(options).run();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
