#include "meshaware/coefficient_patterns.hpp"
#include "meshaware/driver_cli.hpp"
#include "meshaware/experiment_record.hpp"
#include "meshaware/petsc_amg.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace dealii;
using Clock = std::chrono::steady_clock;

constexpr char mesh_family_name[] = "simplex-dg";
constexpr char matrix_prefix[] = "simplex_dg";

double seconds_since(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Options {
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
  bool verify = false;
};

Options parse_options(const int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--mesh-family") {
      const std::string family =
          meshaware::require_option_value(i, argc, argv);
      if (family != mesh_family_name)
        throw std::invalid_argument(
            "simplex DG mesh family must be 'simplex-dg'");
    } else if (argument == "--level")
      options.level =
          std::stoul(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--epsilon")
      options.epsilon =
          std::stod(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--pattern")
      options.pattern = meshaware::parse_pattern(
          meshaware::require_option_value(i, argc, argv));
    else if (argument == "--high-region")
      options.high_region = meshaware::parse_high_region(
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
      options.amg_smoother = meshaware::parse_amg_smoother(
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
    else if (argument == "--verify")
      options.verify = true;
    else if (argument == "--help") {
      std::cout
          << "Usage: meshaware_diffusion_simplex_dg [options]\n"
          << "  --mesh-family NAME     simplex-dg\n"
          << "  --level N              nominal h = 2^-N\n"
          << "  --epsilon E            coefficient contrast = 10^E\n"
          << "  --pattern NAME         vertical_split, checkerboard_2x2,\n"
          << "                         vertical_stripes_4, checkerboard_4x4\n"
          << "  --high-region NAME     white or gray\n"
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
          << "  --assemble-only         export matrix without solving\n"
          << "  --verify                check SIPG matrix symmetry\n";
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

struct ErrorNorms {
  double l2;
  double broken_h1;
  double energy;
};

class SimplexDgExperiment {
public:
  explicit SimplexDgExperiment(const Options &options)
      : options(options), finite_element(1), dof_handler(triangulation) {
    constraints.close();
  }

  void run() {
    make_grid();
    setup_system();

    const auto assembly_start = Clock::now();
    assemble_system();
    const double assembly_seconds = seconds_since(assembly_start);

    if (options.verify)
      verify_matrix_symmetry();

    const std::uint64_t nonzeros = matrix_nonzeros();
    if (!options.matrix_path.empty() && !options.skip_matrix_write)
      meshaware::write_petsc_matrix(static_cast<Mat>(system_matrix),
                                    options.matrix_path);

    if (options.assemble_only) {
      std::cout << "assembled_only=1 family=" << mesh_family_name
                << " cells=" << triangulation.n_active_cells()
                << " dofs=" << dof_handler.n_dofs()
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
  void make_grid() {
    const unsigned int subdivisions = 1u << (options.level + 1);
    Triangulation<2> quadrilateral_grid;
    GridGenerator::subdivided_hyper_rectangle(
        quadrilateral_grid, std::vector<unsigned int>(2, subdivisions),
        Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), false);
    GridGenerator::convert_hypercube_to_simplex_mesh(quadrilateral_grid,
                                                      triangulation, 2);

    h_max = 0.0;
    for (const auto &cell : triangulation.active_cell_iterators())
      h_max = std::max(h_max, cell->diameter());
  }

  void setup_system() {
    dof_handler.distribute_dofs(finite_element);
    DynamicSparsityPattern dynamic_pattern(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dynamic_pattern);
    sparsity_pattern.copy_from(dynamic_pattern);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(PETSC_COMM_SELF, dof_handler.n_dofs(),
                    dof_handler.n_dofs());
    right_hand_side.reinit(PETSC_COMM_SELF, dof_handler.n_dofs(),
                           dof_handler.n_dofs());
  }

  double coefficient(const Point<2> &point) const {
    return meshaware::diffusion_coefficient(options.pattern, options.epsilon,
                                            point[0], point[1],
                                            options.high_region);
  }

  double forcing(const Point<2> &point) const {
    return meshaware::forcing_value(options.pattern, options.epsilon, point[0],
                                    point[1], options.high_region);
  }

  double exact_value(const Point<2> &point) const {
    return meshaware::exact_value(options.pattern, point[0], point[1]);
  }

  double cell_coefficient(
      const DoFHandler<2>::cell_iterator &cell) const {
    return coefficient(cell->center());
  }

  double penalty(const double mu_0, const double h_0, const double mu_1,
                 const double h_1) const {
    const double penalty_constant =
        10.0 * (finite_element.degree + 1.0) *
        (finite_element.degree + 2.0);
    return penalty_constant * std::max(mu_0 / h_0, mu_1 / h_1);
  }

  void assemble_system() {
    const QGaussSimplex<2> cell_quadrature(finite_element.degree + 1);
    const QGauss<1> face_quadrature(finite_element.degree + 1);
    FEValues<2> cell_values(
        finite_element, cell_quadrature,
        update_values | update_gradients | update_quadrature_points |
            update_JxW_values);
    FEInterfaceValues<2> interface_values(
        finite_element, face_quadrature,
        update_values | update_gradients | update_quadrature_points |
            update_normal_vectors | update_JxW_values);

    const unsigned int dofs_per_cell = finite_element.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      cell_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs = 0.0;
      for (const unsigned int q : cell_values.quadrature_point_indices()) {
        const Point<2> &point = cell_values.quadrature_point(q);
        const double mu = coefficient(point);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          cell_rhs(i) += cell_values.shape_value(i, q) * forcing(point) *
                         cell_values.JxW(q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += mu * cell_values.shape_grad(i, q) *
                                 cell_values.shape_grad(j, q) *
                                 cell_values.JxW(q);
        }
      }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix,
          right_hand_side);
    }

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      const double mu_0 = cell_coefficient(cell);
      const double h_0 = cell->diameter();
      for (const unsigned int face : cell->face_indices()) {
        if (cell->at_boundary(face)) {
          interface_values.reinit(cell, face);
          const double sigma = penalty(mu_0, h_0, mu_0, h_0);
          assemble_boundary_face(interface_values, mu_0, sigma);
          continue;
        }

        const auto neighbor = cell->neighbor(face);
        if (cell->active_cell_index() >= neighbor->active_cell_index())
          continue;
        const unsigned int neighbor_face = cell->neighbor_of_neighbor(face);
        interface_values.reinit(
            cell, face, numbers::invalid_unsigned_int, neighbor,
            neighbor_face, numbers::invalid_unsigned_int);
        const double mu_1 = cell_coefficient(neighbor);
        const double sigma =
            penalty(mu_0, h_0, mu_1, neighbor->diameter());
        assemble_interior_face(interface_values, mu_0, mu_1, sigma);
      }
    }

    system_matrix.compress(VectorOperation::add);
    right_hand_side.compress(VectorOperation::add);
  }

  void assemble_boundary_face(FEInterfaceValues<2> &interface_values,
                              const double mu, const double sigma) {
    const unsigned int n_dofs = interface_values.n_current_interface_dofs();
    FullMatrix<double> face_matrix(n_dofs, n_dofs);
    Vector<double> face_rhs(n_dofs);
    const auto global_dof_indices = interface_values.get_interface_dof_indices();

    for (const unsigned int q : interface_values.quadrature_point_indices()) {
      const Tensor<1, 2> normal = interface_values.normal_vector(q);
      const double boundary_value =
          exact_value(interface_values.quadrature_point(q));
      for (const unsigned int i : interface_values.dof_indices()) {
        const double value_i =
            interface_values.jump_in_shape_values(i, q);
        const double flux_i =
            mu * (interface_values.shape_grad(true, i, q) * normal);
        face_rhs(i) += (-flux_i * boundary_value +
                        sigma * boundary_value * value_i) *
                       interface_values.JxW(q);
        for (const unsigned int j : interface_values.dof_indices()) {
          const double value_j =
              interface_values.jump_in_shape_values(j, q);
          const double flux_j =
              mu * (interface_values.shape_grad(true, j, q) * normal);
          face_matrix(i, j) +=
              (-flux_i * value_j - flux_j * value_i +
               sigma * value_i * value_j) *
              interface_values.JxW(q);
        }
      }
    }
    constraints.distribute_local_to_global(
        face_matrix, face_rhs, global_dof_indices, system_matrix,
        right_hand_side);
  }

  void assemble_interior_face(FEInterfaceValues<2> &interface_values,
                              const double mu_0, const double mu_1,
                              const double sigma) {
    const unsigned int n_dofs = interface_values.n_current_interface_dofs();
    FullMatrix<double> face_matrix(n_dofs, n_dofs);
    const auto global_dof_indices = interface_values.get_interface_dof_indices();

    for (const unsigned int q : interface_values.quadrature_point_indices()) {
      const Tensor<1, 2> normal = interface_values.normal_vector(q);
      for (const unsigned int i : interface_values.dof_indices()) {
        const double jump_i =
            interface_values.jump_in_shape_values(i, q);
        const double average_flux_i =
            0.5 * (mu_0 * interface_values.shape_grad(true, i, q) +
                   mu_1 * interface_values.shape_grad(false, i, q)) *
            normal;
        for (const unsigned int j : interface_values.dof_indices()) {
          const double jump_j =
              interface_values.jump_in_shape_values(j, q);
          const double average_flux_j =
              0.5 * (mu_0 * interface_values.shape_grad(true, j, q) +
                     mu_1 * interface_values.shape_grad(false, j, q)) *
              normal;
          face_matrix(i, j) +=
              (-average_flux_i * jump_j - average_flux_j * jump_i +
               sigma * jump_i * jump_j) *
              interface_values.JxW(q);
        }
      }
    }
    constraints.distribute_local_to_global(face_matrix, global_dof_indices,
                                           system_matrix);
  }

  void verify_matrix_symmetry() const {
    const Mat matrix = static_cast<Mat>(system_matrix);
    Mat transpose_difference = nullptr;
    meshaware::petsc_check(
        MatTranspose(matrix, MAT_INITIAL_MATRIX, &transpose_difference),
        "MatTranspose(simplex SIPG)");
    try {
      meshaware::petsc_check(
          MatAXPY(transpose_difference, -1.0, matrix, SAME_NONZERO_PATTERN),
          "MatAXPY(simplex SIPG symmetry defect)");
      PetscReal matrix_norm = 0.0;
      PetscReal defect_norm = 0.0;
      meshaware::petsc_check(MatNorm(matrix, NORM_INFINITY, &matrix_norm),
                             "MatNorm(simplex SIPG)");
      meshaware::petsc_check(
          MatNorm(transpose_difference, NORM_INFINITY, &defect_norm),
          "MatNorm(simplex SIPG symmetry defect)");
      const double relative_defect =
          matrix_norm == 0.0 ? defect_norm : defect_norm / matrix_norm;
      if (relative_defect > 1e-12)
        throw std::runtime_error(
            "simplex SIPG relative symmetry defect exceeds tolerance: " +
            std::to_string(relative_defect));
    } catch (...) {
      MatDestroy(&transpose_difference);
      throw;
    }
    meshaware::petsc_check(MatDestroy(&transpose_difference),
                           "MatDestroy(simplex SIPG transpose)");
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
    return meshaware::solve_with_boomer_amg(
        static_cast<Mat>(system_matrix),
        static_cast<const Vec &>(right_hand_side),
        static_cast<const Vec &>(solution), solver_options);
  }

  ErrorNorms compute_errors() const {
    const QGaussSimplex<2> cell_quadrature(finite_element.degree + 2);
    const QGauss<1> face_quadrature(finite_element.degree + 2);
    FEValues<2> cell_values(
        finite_element, cell_quadrature,
        update_values | update_gradients | update_quadrature_points |
            update_JxW_values);
    FEInterfaceValues<2> interface_values(
        finite_element, face_quadrature,
        update_values | update_quadrature_points | update_JxW_values);

    std::vector<double> values(cell_quadrature.size());
    std::vector<Tensor<1, 2>> gradients(cell_quadrature.size());
    double l2_squared = 0.0;
    double broken_h1_squared = 0.0;
    double energy_squared = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      cell_values.reinit(cell);
      cell_values.get_function_values(solution, values);
      cell_values.get_function_gradients(solution, gradients);
      for (const unsigned int q : cell_values.quadrature_point_indices()) {
        const Point<2> &point = cell_values.quadrature_point(q);
        const double value_error = values[q] - exact_value(point);
        const auto exact_gradient =
            meshaware::exact_gradient(options.pattern, point[0], point[1]);
        Tensor<1, 2> gradient_error = gradients[q];
        gradient_error[0] -= exact_gradient[0];
        gradient_error[1] -= exact_gradient[1];
        const double weight = cell_values.JxW(q);
        l2_squared += value_error * value_error * weight;
        broken_h1_squared += gradient_error.norm_square() * weight;
        energy_squared +=
            coefficient(point) * gradient_error.norm_square() * weight;
      }
    }

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      const double mu_0 = cell_coefficient(cell);
      const double h_0 = cell->diameter();
      for (const unsigned int face : cell->face_indices()) {
        double sigma = 0.0;
        if (cell->at_boundary(face)) {
          interface_values.reinit(cell, face);
          sigma = penalty(mu_0, h_0, mu_0, h_0);
        } else {
          const auto neighbor = cell->neighbor(face);
          if (cell->active_cell_index() >= neighbor->active_cell_index())
            continue;
          const unsigned int neighbor_face = cell->neighbor_of_neighbor(face);
          interface_values.reinit(
              cell, face, numbers::invalid_unsigned_int, neighbor,
              neighbor_face, numbers::invalid_unsigned_int);
          sigma = penalty(mu_0, h_0, cell_coefficient(neighbor),
                          neighbor->diameter());
        }

        const auto global_dof_indices =
            interface_values.get_interface_dof_indices();
        for (const unsigned int q :
             interface_values.quadrature_point_indices()) {
          double jump = cell->at_boundary(face)
                            ? -exact_value(interface_values.quadrature_point(q))
                            : 0.0;
          for (const unsigned int i : interface_values.dof_indices())
            jump += solution[global_dof_indices[i]] *
                    interface_values.jump_in_shape_values(i, q);
          energy_squared +=
              sigma * jump * jump * interface_values.JxW(q);
        }
      }
    }

    return {std::sqrt(l2_squared), std::sqrt(broken_h1_squared),
            std::sqrt(energy_squared)};
  }

  std::string matrix_id() const {
    return meshaware::make_matrix_id(
        matrix_prefix, options.level, meshaware::to_string(options.pattern),
        options.epsilon, meshaware::to_string(options.high_region));
  }

  std::filesystem::path batch_record_path(const double theta,
                                          const unsigned int repeat) const {
    return options.record_directory /
           (meshaware::make_sample_id(matrix_id(), theta, repeat) + ".json");
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
    record.mesh_family = mesh_family_name;
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
    record.h1_seminorm_error = errors.broken_h1;
    record.energy_error = errors.energy;
    record.assembly_time_seconds = assembly_seconds;
    record.amg_setup_time_seconds = solver_metrics.setup_seconds;
    record.solve_time_seconds = solver_metrics.solve_seconds;
    record.matrix_path = options.matrix_path;
    meshaware::write_experiment_record(record, record_path);
  }

  void run_trial(const double theta, const unsigned int repeat,
                 const std::filesystem::path &record_path,
                 const double assembly_seconds,
                 const std::uint64_t nonzeros) {
    const meshaware::SolverMetrics solver_metrics = solve(theta);
    const ErrorNorms errors = compute_errors();
    if (!std::isfinite(errors.l2) || !std::isfinite(errors.broken_h1) ||
        !std::isfinite(errors.energy))
      throw std::runtime_error("simplex SIPG produced a non-finite error");
    write_record(theta, repeat, record_path, assembly_seconds, solver_metrics,
                 errors, nonzeros);

    std::cout << std::setprecision(8)
              << "simplex_dg cells=" << triangulation.n_active_cells()
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
              << solver_metrics.operator_complexity
              << " l2_error=" << errors.l2
              << " broken_h1_error=" << errors.broken_h1
              << " energy_error=" << errors.energy << '\n';
  }

  void run_batch(const double assembly_seconds,
                 const std::uint64_t nonzeros) {
    unsigned int completed = 0;
    unsigned int skipped = 0;
    for (const double theta : options.theta_values) {
      std::vector<unsigned int> pending_repeats;
      for (unsigned int repeat = 0; repeat < options.repeats; ++repeat) {
        const auto path = batch_record_path(theta, repeat);
        if (options.skip_existing_records && std::filesystem::exists(path))
          ++skipped;
        else
          pending_repeats.push_back(repeat);
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
    std::cout << "batch_completed=" << completed
              << " batch_skipped=" << skipped << '\n';
  }

  const Options options;
  Triangulation<2> triangulation;
  FE_SimplexDGP<2> finite_element;
  DoFHandler<2> dof_handler;
  AffineConstraints<double> constraints;
  SparsityPattern sparsity_pattern;
  PETScWrappers::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector solution;
  PETScWrappers::MPI::Vector right_hand_side;
  double h_max = 0.0;
};
} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    int initialization_argc = 1;
    char *initialization_arguments[] = {argv[0], nullptr};
    char **initialization_argv = initialization_arguments;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(initialization_argc,
                                                        initialization_argv, 1);
    SimplexDgExperiment(options).run();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
