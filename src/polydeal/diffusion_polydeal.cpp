#include "meshaware/coefficient_patterns.hpp"
#include "meshaware/driver_cli.hpp"
#include "meshaware/experiment_record.hpp"
#include "meshaware/petsc_amg.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

#include <agglomeration_handler.h>
#include <fe_agglodgp.h>
#include <multigrid_amg.h>
#include <poly_utils.h>
#include <utils.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
using namespace dealii;
using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Options {
  unsigned int level = 2;
  meshaware::Pattern pattern = meshaware::Pattern::vertical_split;
  double epsilon = 0.0;
  meshaware::HighRegion high_region = meshaware::HighRegion::white;
  double theta = 0.24;
  double relative_tolerance = 1e-8;
  double absolute_tolerance = 1e-50;
  unsigned int maximum_iterations = 10000;
  meshaware::AmgBackend amg_backend = meshaware::AmgBackend::boomeramg;
  meshaware::BoomerAmgProfile boomeramg_profile =
      meshaware::BoomerAmgProfile::default_options;
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
  bool oracle = false;
  bool assemble_only = false;
};

Options parse_options(const int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--mesh-family") {
      const std::string family =
          meshaware::require_option_value(i, argc, argv);
      if (family != "polygonal")
        throw std::invalid_argument("PolyDeal mesh family must be 'polygonal'");
    } else if (argument == "--level")
      options.level =
          std::stoul(meshaware::require_option_value(i, argc, argv));
    else if (argument == "--pattern")
      options.pattern = meshaware::parse_pattern(
          meshaware::require_option_value(i, argc, argv));
    else if (argument == "--epsilon")
      options.epsilon =
          std::stod(meshaware::require_option_value(i, argc, argv));
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
    else if (argument == "--amg-backend")
      options.amg_backend =
          meshaware::parse_amg_backend(
              meshaware::require_option_value(i, argc, argv));
    else if (argument == "--boomeramg-profile")
      options.boomeramg_profile =
          meshaware::parse_boomeramg_profile(
              meshaware::require_option_value(i, argc, argv));
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
    else if (argument == "--oracle")
      options.oracle = true;
    else if (argument == "--assemble-only")
      options.assemble_only = true;
    else if (argument == "--help") {
      std::cout
          << "Usage: meshaware_diffusion_polydeal [options]\n"
          << "  --mesh-family polygonal\n"
          << "  --level N       nominal background h = 2^-N\n"
          << "  --pattern NAME  manufactured-solution pattern\n"
          << "  --epsilon E     coefficient contrast = 10^E\n"
          << "  --high-region NAME  white or gray\n"
          << "  --theta T       BoomerAMG strong threshold; recorded but "
             "unused by PolyDeal MG\n"
          << "  --theta-values CSV  batched strong-threshold grid\n"
          << "  --rtol T --atol T --max-iterations N\n"
          << "  --amg-backend NAME boomeramg or "
             "polydeal-agglomeration\n"
          << "  --boomeramg-profile NAME default or polygonal-nodal\n"
          << "  --amg-smoother NAME chebyshev, damped-jacobi, "
             "l1-symmetric-gauss-seidel, or symmetric-gauss-seidel\n"
          << "  --jacobi-damping W  damping in (0,1], default 2/3\n"
          << "  --repeat N      timing repeat identifier\n"
          << "  --repeats N     repeats per theta in batch mode\n"
          << "  --warmup-runs N discarded warm-ups per theta\n"
          << "  --record PATH   JSON trial record\n"
          << "  --record-dir PATH  generated records for batch mode\n"
          << "  --matrix PATH   PETSc binary matrix/reference\n"
          << "  --skip-matrix-write\n"
          << "  --skip-existing-records\n"
          << "  --oracle        duplicate into native matrix and compare\n"
          << "  --assemble-only export matrix without solving\n";
      std::exit(0);
    } else
      throw std::invalid_argument("Unknown argument: " + argument);
  }

  if (options.level > 10)
    throw std::invalid_argument("level must not exceed 10");
  if (options.epsilon < 0.0)
    throw std::invalid_argument("epsilon must be non-negative");
  if (options.theta_values.empty())
    options.theta_values.push_back(options.theta);
  if (std::any_of(
          options.theta_values.begin(), options.theta_values.end(),
          [](const double theta) { return !(theta > 0.0 && theta < 1.0); }))
    throw std::invalid_argument("all theta values must lie strictly in (0,1)");
  if (options.relative_tolerance <= 0.0 || options.absolute_tolerance < 0.0 ||
      options.maximum_iterations == 0)
    throw std::invalid_argument("invalid solver tolerances or iteration limit");
  if (!(options.jacobi_damping > 0.0 && options.jacobi_damping <= 1.0))
    throw std::invalid_argument("jacobi-damping must lie in (0,1]");
  if (options.repeats == 0)
    throw std::invalid_argument("repeats must be positive");
  if (options.amg_backend ==
          meshaware::AmgBackend::polydeal_agglomeration &&
      options.theta_values.size() != 1)
    throw std::invalid_argument(
        "polydeal-agglomeration accepts one theta value because it does not "
        "use a strength threshold");
  if (options.amg_backend ==
          meshaware::AmgBackend::polydeal_agglomeration &&
      options.boomeramg_profile !=
          meshaware::BoomerAmgProfile::default_options)
    throw std::invalid_argument(
        "BoomerAMG tuning profiles require the boomeramg backend");
  if (options.amg_backend ==
          meshaware::AmgBackend::polydeal_agglomeration &&
      options.amg_smoother ==
          meshaware::AmgSmoother::l1_symmetric_gauss_seidel)
    throw std::invalid_argument(
        "l1-symmetric-gauss-seidel is available only for BoomerAMG");
  if (options.boomeramg_profile ==
          meshaware::BoomerAmgProfile::polygonal_nodal &&
      options.amg_smoother ==
          meshaware::AmgSmoother::l1_symmetric_gauss_seidel)
    throw std::invalid_argument(
        "polygonal-nodal is incompatible with HYPRE's "
        "l1-symmetric-gauss-seidel relaxation");
  if (!options.record_directory.empty() && !options.record_path.empty())
    throw std::invalid_argument("record and record-dir are mutually exclusive");
  if (options.record_directory.empty() && options.theta_values.size() != 1)
    throw std::invalid_argument("theta-values requires record-dir batch mode");
  if (options.oracle && !options.record_directory.empty())
    throw std::invalid_argument("oracle mode does not support batch execution");
  if (options.oracle && options.level > 6)
    throw std::invalid_argument("oracle mode is restricted to level <= 6");
  if (options.assemble_only &&
      (options.matrix_path.empty() || options.skip_matrix_write))
    throw std::invalid_argument(
        "assemble-only requires a writable --matrix path");
  if (options.assemble_only &&
      (!options.record_path.empty() || !options.record_directory.empty()))
    throw std::invalid_argument(
        "assemble-only does not accept record output options");
  if (options.assemble_only && options.oracle)
    throw std::invalid_argument("assemble-only and oracle are incompatible");
  return options;
}

class ExactSolution : public Function<2> {
public:
  explicit ExactSolution(const meshaware::Pattern pattern)
      : Function<2>(1), pattern(pattern) {}

  double value(const Point<2> &point,
               const unsigned int component = 0) const override {
    (void)component;
    return meshaware::exact_value(pattern, point[0], point[1]);
  }

  Tensor<1, 2> gradient(const Point<2> &point,
                        const unsigned int component = 0) const override {
    (void)component;
    const auto exact = meshaware::exact_gradient(pattern, point[0], point[1]);
    Tensor<1, 2> result;
    result[0] = exact[0];
    result[1] = exact[1];
    return result;
  }

private:
  meshaware::Pattern pattern;
};

struct ErrorNorms {
  double l2 = 0.0;
  double broken_h1 = 0.0;
  double energy = 0.0;
};

using BackgroundCell = Triangulation<2>::active_cell_iterator;

struct AgglomerateGroup {
  std::vector<BackgroundCell> cells;
  std::vector<unsigned int> children;
  unsigned int tile_x = 0;
  unsigned int tile_y = 0;
  unsigned int grid_x = 0;
  unsigned int grid_y = 0;
};

void fill_nested_injection_matrix(
    const AgglomerationHandler<2> &coarse_handler,
    const AgglomerationHandler<2> &fine_handler,
    const std::vector<std::vector<unsigned int>> &children_by_parent,
    TrilinosWrappers::SparseMatrix &injection) {
  if (coarse_handler.n_dofs() >= fine_handler.n_dofs())
    throw std::runtime_error(
        "agglomeration hierarchy does not reduce the number of DoFs");
  if (children_by_parent.size() != coarse_handler.n_agglomerates())
    throw std::runtime_error("invalid parent-child map for polygon hierarchy");

  const auto &coarse_dof_handler = coarse_handler.agglo_dh;
  const auto &fine_dof_handler = fine_handler.agglo_dh;
  const auto &finite_element = coarse_handler.get_fe();
  const auto &fine_bboxes = fine_handler.get_local_bboxes();
  const auto &coarse_bboxes = coarse_handler.get_local_bboxes();
  const MPI_Comm communicator = coarse_dof_handler.get_mpi_communicator();

  TrilinosWrappers::SparsityPattern sparsity(
      fine_dof_handler.locally_owned_dofs(),
      coarse_dof_handler.locally_owned_dofs(), communicator);
  const unsigned int dofs_per_cell = finite_element.dofs_per_cell;
  std::vector<types::global_dof_index> coarse_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> fine_dof_indices(dofs_per_cell);

  for (const auto &coarse_polytope : coarse_handler.polytope_iterators()) {
    coarse_polytope->get_dof_indices(coarse_dof_indices);
    const auto &children = children_by_parent.at(coarse_polytope->index());
    if (children.empty())
      throw std::runtime_error("coarse polygon has no fine children");
    for (const unsigned int child_index : children) {
      const auto &child =
          fine_handler.polytope_to_dh_iterator(child_index);
      child->get_dof_indices(fine_dof_indices);
      for (const auto row : fine_dof_indices)
        sparsity.add_entries(row, coarse_dof_indices.begin(),
                             coarse_dof_indices.end());
    }
  }

  sparsity.compress();
  injection.reinit(sparsity);

  // FE_AggloDGP is modal and has no support points. Recover the exact
  // polynomial restriction by evaluating both bases at an unisolvent set.
  std::vector<Point<2>> interpolation_points;
  const unsigned int degree = finite_element.degree;
  if (degree == 0) {
    interpolation_points.emplace_back(0.5, 0.5);
  } else {
    for (unsigned int i = 0; i <= degree; ++i)
      for (unsigned int j = 0; j + i <= degree; ++j)
        interpolation_points.emplace_back(
            static_cast<double>(i) / degree,
            static_cast<double>(j) / degree);
  }
  if (interpolation_points.size() != dofs_per_cell)
    throw std::runtime_error(
        "cannot construct an unisolvent PolyDeal transfer basis");

  FullMatrix<double> fine_vandermonde(dofs_per_cell, dofs_per_cell);
  for (unsigned int i = 0; i < dofs_per_cell; ++i)
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
      fine_vandermonde(i, j) =
          finite_element.shape_value(j, interpolation_points[i]);
  FullMatrix<double> inverse_fine_vandermonde(dofs_per_cell, dofs_per_cell);
  inverse_fine_vandermonde.invert(fine_vandermonde);

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> coarse_values(dofs_per_cell, dofs_per_cell);
  AffineConstraints<double> no_constraints;
  no_constraints.close();
  for (const auto &coarse_polytope : coarse_handler.polytope_iterators()) {
    coarse_polytope->get_dof_indices(coarse_dof_indices);
    const BoundingBox<2> &coarse_bbox =
        coarse_bboxes.at(coarse_polytope->index());

    for (const unsigned int child_index :
         children_by_parent.at(coarse_polytope->index())) {
      const BoundingBox<2> &fine_bbox = fine_bboxes.at(child_index);
      const auto &child =
          fine_handler.polytope_to_dh_iterator(child_index);
      child->get_dof_indices(fine_dof_indices);
      coarse_values = 0.0;

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        const Point<2> real_point =
            fine_bbox.unit_to_real(interpolation_points[i]);
        const Point<2> coarse_point = coarse_bbox.real_to_unit(real_point);
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          coarse_values(i, j) =
              finite_element.shape_value(j, coarse_point);
      }
      inverse_fine_vandermonde.mmult(local_matrix, coarse_values);

      no_constraints.distribute_local_to_global(
          local_matrix, fine_dof_indices, coarse_dof_indices, injection);
    }
  }
  injection.compress(VectorOperation::add);
}

class PolyDealExperiment {
public:
  explicit PolyDealExperiment(const Options &options)
      : options(options), finite_element(1) {
    constraints.close();
  }

  void run() {
    make_grid_and_agglomerates();
    setup_system();

    const auto assembly_start = Clock::now();
    assemble_system();
    const double assembly_seconds = seconds_since(assembly_start);

    if (options.boomeramg_profile ==
        meshaware::BoomerAmgProfile::polygonal_nodal)
      prepare_polygonal_nodal_boomeramg();

    const std::uint64_t nonzeros = matrix_nonzeros();
    if (!options.matrix_path.empty() && !options.skip_matrix_write)
      meshaware::write_petsc_matrix(static_cast<Mat>(petsc_matrix),
                                    options.matrix_path);

    if (options.assemble_only) {
      std::cout << "assembled_only=1 dofs="
                << agglomeration_handler->n_dofs()
                << " nnz=" << nonzeros << '\n';
      return;
    }

    if (!options.record_directory.empty()) {
      run_batch(assembly_seconds, nonzeros);
      return;
    }

    double symmetry_defect = 0.0;
    double matrix_defect = 0.0;
    double rhs_defect = 0.0;
    if (options.oracle) {
      symmetry_defect = matrix_symmetry_defect();
      matrix_defect = matrix_equivalence_defect();
      rhs_defect = rhs_equivalence_defect();
      solve_direct();
    }

    const double theta = options.theta_values.front();
    const meshaware::SolverMetrics solver_metrics = solve_system(theta);
    const double solution_defect =
        options.oracle ? solution_equivalence_defect() : 0.0;
    const ErrorNorms errors = compute_errors(petsc_solution_native);

    std::cout << (options.oracle ? "polydeal_oracle" : "polydeal")
              << " background_cells=" << triangulation.n_active_cells()
              << " polygons=" << agglomeration_handler->n_agglomerates()
              << " dofs=" << agglomeration_handler->n_dofs()
              << " h_max=" << h_max << " nnz=" << nonzeros
              << " symmetry_defect=" << symmetry_defect
              << " matrix_defect=" << matrix_defect
              << " rhs_defect=" << rhs_defect
              << " solution_defect=" << solution_defect
              << " backend=" << meshaware::to_string(options.amg_backend)
              << " boomeramg_profile="
              << meshaware::to_string(options.boomeramg_profile)
              << " smoother=" << meshaware::to_string(options.amg_smoother)
              << " cg_iterations=" << solver_metrics.iterations
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

    if (!std::isfinite(errors.l2) || !std::isfinite(errors.broken_h1) ||
        !std::isfinite(errors.energy))
      throw std::runtime_error("PolyDeal produced a non-finite error");
    if (options.oracle) {
      if (symmetry_defect > 1e-13)
        throw std::runtime_error("SIPG matrix is not symmetric");
      if (matrix_defect > 1e-12 || rhs_defect > 1e-12)
        throw std::runtime_error(
            "native and PETSc SIPG assembly results do not match");
      if (options.epsilon <= 2.0 && solution_defect > 1e-7)
        throw std::runtime_error(
            "native direct and PETSc iterative solutions do not match");
    }

    write_record(theta, options.repeat, options.record_path, assembly_seconds,
                 solver_metrics, errors, nonzeros);
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
        (void)solve_system(theta);

      for (const unsigned int repeat : pending_repeats) {
        const meshaware::SolverMetrics solver_metrics = solve_system(theta);
        const ErrorNorms errors = compute_errors(petsc_solution_native);
        write_record(theta, repeat, batch_record_path(theta, repeat),
                     assembly_seconds, solver_metrics, errors, nonzeros);
        std::cout << "polydeal"
                  << " polygons=" << agglomeration_handler->n_agglomerates()
                  << " dofs=" << agglomeration_handler->n_dofs()
                  << " theta=" << theta << " repeat=" << repeat
                  << " backend=" << meshaware::to_string(options.amg_backend)
                  << " boomeramg_profile="
                  << meshaware::to_string(options.boomeramg_profile)
                  << " smoother="
                  << meshaware::to_string(options.amg_smoother)
                  << " iterations=" << solver_metrics.iterations
                  << " amg_levels=" << solver_metrics.amg_levels
                  << " rho=" << solver_metrics.convergence_factor
                  << " setup_s=" << solver_metrics.setup_seconds
                  << " solve_s=" << solver_metrics.solve_seconds
                  << " grid_complexity=" << solver_metrics.grid_complexity
                  << " operator_complexity="
                  << solver_metrics.operator_complexity << '\n';
        ++completed;
      }
    }
    std::cout << "batch_completed=" << completed << " batch_skipped=" << skipped
              << '\n';
  }

  void make_grid_and_agglomerates() {
    const unsigned int subdivisions = 1u << (options.level + 1);
    GridGenerator::subdivided_hyper_rectangle(
        triangulation, std::vector<unsigned int>(2, subdivisions),
        Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), false);

    cached_triangulation =
        std::make_unique<GridTools::Cache<2>>(triangulation, mapping);
    build_tile_constrained_hierarchy(subdivisions);
  }

  void
  build_tile_constrained_hierarchy(const unsigned int subdivisions) {
    using AgglomerateKey = std::tuple<unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int>;
    std::map<AgglomerateKey, std::vector<BackgroundCell>> fine_cells;

    // Use the common finest interface layout for every pattern. This keeps
    // polygon geometry and DoF ordering independent of the coefficient
    // pattern while ensuring no agglomerate crosses any of the four fields.
    const std::array<unsigned int, 2> counts{{4, 4}};
    if (subdivisions % counts[0] != 0 || subdivisions % counts[1] != 0)
      throw std::runtime_error(
          "background grid does not align with coefficient tiles");
    const unsigned int tile_width = subdivisions / counts[0];
    const unsigned int tile_height = subdivisions / counts[1];
    const double background_h = 2.0 / subdivisions;

    for (const auto &cell : triangulation.active_cell_iterators()) {
      const Point<2> center = cell->center();
      const unsigned int column =
          std::min(static_cast<unsigned int>(
                       std::floor((center[0] + 1.0) / background_h)),
                   subdivisions - 1);
      const unsigned int row = std::min(static_cast<unsigned int>(std::floor(
                                            (center[1] + 1.0) / background_h)),
                                        subdivisions - 1);
      const unsigned int tile_x = column / tile_width;
      const unsigned int tile_y = row / tile_height;
      const unsigned int local_x = column % tile_width;
      const unsigned int local_y = row % tile_height;
      const unsigned int block_x = local_x / 2;
      const unsigned int block_y = local_y / 4;
      const unsigned int offset_x = local_x % 2;
      const unsigned int offset_y = local_y % 4;

      // Each complete 2x4 background block is tiled by two connected,
      // four-cell L polyominoes. Truncated blocks at very coarse tile
      // resolutions remain connected and never cross a coefficient tile.
      const bool first_polyomino =
          (offset_x == 0 && offset_y == 0) || (offset_x == 1 && offset_y <= 2);
      fine_cells[{tile_x, tile_y, block_x, block_y,
                  first_polyomino ? 0u : 1u}]
          .push_back(cell);
    }

    std::vector<std::vector<AgglomerateGroup>> partitions_fine_to_coarse(1);
    auto &finest_partition = partitions_fine_to_coarse.front();
    finest_partition.reserve(fine_cells.size());
    for (const auto &[key, cells] : fine_cells) {
      if (cells.empty())
        throw std::runtime_error("constructed an empty agglomerate");
      const auto [tile_x, tile_y, block_x, block_y, fine_part] = key;
      (void)fine_part;
      finest_partition.push_back(
          {cells, {}, tile_x, tile_y, block_x, block_y});
    }

    bool merge_fine_parts = true;
    while (partitions_fine_to_coarse.back().size() > 16) {
      const auto &children = partitions_fine_to_coarse.back();
      using ParentKey =
          std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>;
      std::map<ParentKey, AgglomerateGroup> parents;

      for (unsigned int child_index = 0; child_index < children.size();
           ++child_index) {
        const auto &child = children[child_index];
        const unsigned int parent_x =
            merge_fine_parts ? child.grid_x : child.grid_x / 2;
        const unsigned int parent_y =
            merge_fine_parts ? child.grid_y : child.grid_y / 2;
        auto &parent =
            parents[{child.tile_x, child.tile_y, parent_x, parent_y}];
        parent.tile_x = child.tile_x;
        parent.tile_y = child.tile_y;
        parent.grid_x = parent_x;
        parent.grid_y = parent_y;
        parent.cells.insert(parent.cells.end(), child.cells.begin(),
                            child.cells.end());
        parent.children.push_back(child_index);
      }

      std::vector<AgglomerateGroup> next_partition;
      next_partition.reserve(parents.size());
      for (auto &[key, parent] : parents) {
        (void)key;
        next_partition.push_back(std::move(parent));
      }
      if (next_partition.size() >= children.size())
        break;
      partitions_fine_to_coarse.push_back(std::move(next_partition));
      merge_fine_parts = false;
    }

    mg_agglomeration_handlers.clear();
    mg_level_children.clear();
    mg_agglomeration_handlers.reserve(partitions_fine_to_coarse.size());
    if (partitions_fine_to_coarse.size() > 1)
      mg_level_children.resize(partitions_fine_to_coarse.size() - 1);

    for (unsigned int level = 0; level < partitions_fine_to_coarse.size();
         ++level) {
      const unsigned int partition_index =
          partitions_fine_to_coarse.size() - 1 - level;
      auto handler =
          std::make_unique<AgglomerationHandler<2>>(*cached_triangulation);
      const auto &partition = partitions_fine_to_coarse[partition_index];
      for (const auto &group : partition)
        handler->define_agglomerate(group.cells);
      mg_agglomeration_handlers.push_back(std::move(handler));

      if (level + 1 < partitions_fine_to_coarse.size()) {
        auto &level_children = mg_level_children[level];
        level_children.reserve(partition.size());
        for (const auto &group : partition)
          level_children.push_back(group.children);
      }
    }

    agglomeration_handler = mg_agglomeration_handlers.back().get();
  }

  void setup_system() {
    if (options.amg_backend == meshaware::AmgBackend::polydeal_agglomeration &&
        mg_agglomeration_handlers.size() < 2)
      throw std::runtime_error(
          "polydeal-agglomeration requires at least two polygon levels; "
          "use --level 2 or finer");

    for (auto &handler : mg_agglomeration_handlers)
      handler->distribute_agglomerated_dofs(finite_element);

    agglomeration_handler->create_agglomeration_sparsity_pattern(
        dynamic_sparsity);
    sparsity.copy_from(dynamic_sparsity);
    if (options.oracle) {
      system_matrix.reinit(sparsity);
      right_hand_side.reinit(agglomeration_handler->n_dofs());
      solution.reinit(agglomeration_handler->n_dofs());
    }
    petsc_matrix.reinit(sparsity);
    petsc_right_hand_side.reinit(PETSC_COMM_SELF,
                                 agglomeration_handler->n_dofs(),
                                 agglomeration_handler->n_dofs());
    petsc_solution.reinit(PETSC_COMM_SELF, agglomeration_handler->n_dofs(),
                          agglomeration_handler->n_dofs());
    petsc_solution_native.reinit(agglomeration_handler->n_dofs());

    if (options.amg_backend ==
        meshaware::AmgBackend::polydeal_agglomeration) {
      TrilinosWrappers::SparsityPattern trilinos_sparsity;
      agglomeration_handler->create_agglomeration_sparsity_pattern(
          trilinos_sparsity);
      trilinos_matrix.reinit(trilinos_sparsity);
      const IndexSet &owned_dofs =
          agglomeration_handler->agglo_dh.locally_owned_dofs();
      const MPI_Comm communicator =
          agglomeration_handler->agglo_dh.get_mpi_communicator();
      trilinos_right_hand_side.reinit(owned_dofs, communicator);
      trilinos_solution.reinit(owned_dofs, communicator);
    }

    h_max = 0.0;
    for (const auto &polytope : agglomeration_handler->polytope_iterators())
      h_max = std::max(h_max, std::abs(polytope->diameter()));

    constexpr unsigned int quadrature_degree = 3;
    agglomeration_handler->initialize_fe_values(
        QGauss<2>(quadrature_degree),
        update_values | update_gradients | update_quadrature_points |
            update_JxW_values,
        QGauss<1>(quadrature_degree));
  }

  void prepare_polygonal_nodal_boomeramg() {
    const unsigned int dofs_per_polygon =
        agglomeration_handler->n_dofs_per_cell();
    if (dofs_per_polygon < 2)
      throw std::runtime_error(
          "polygonal-nodal requires multiple modal DoFs per polygon");

    Mat raw_matrix = static_cast<Mat>(petsc_matrix);
    meshaware::petsc_check(
        MatSetBlockSize(raw_matrix, static_cast<PetscInt>(dofs_per_polygon)),
        "MatSetBlockSize(polygonal modal block)");

    // FE_AggloDGP is modal, so an all-ones coefficient vector is not the
    // constant function. Interpolate one on the reference polygon and reuse
    // those coefficients for every agglomerate.
    std::vector<Point<2>> interpolation_points;
    const unsigned int degree = finite_element.degree;
    if (degree == 0) {
      interpolation_points.emplace_back(0.5, 0.5);
    } else {
      for (unsigned int i = 0; i <= degree; ++i)
        for (unsigned int j = 0; j + i <= degree; ++j)
          interpolation_points.emplace_back(
              static_cast<double>(i) / degree,
              static_cast<double>(j) / degree);
    }
    if (interpolation_points.size() != dofs_per_polygon)
      throw std::runtime_error(
          "cannot interpolate the PolyDeal constant near-nullspace mode");

    FullMatrix<double> vandermonde(dofs_per_polygon, dofs_per_polygon);
    for (unsigned int row = 0; row < dofs_per_polygon; ++row)
      for (unsigned int column = 0; column < dofs_per_polygon; ++column)
        vandermonde(row, column) =
            finite_element.shape_value(column, interpolation_points[row]);
    FullMatrix<double> inverse_vandermonde(dofs_per_polygon,
                                            dofs_per_polygon);
    inverse_vandermonde.invert(vandermonde);
    Vector<double> constant_values(dofs_per_polygon);
    Vector<double> constant_coefficients(dofs_per_polygon);
    constant_values = 1.0;
    inverse_vandermonde.vmult(constant_coefficients, constant_values);

    Vec near_nullspace_vector = nullptr;
    meshaware::petsc_check(MatCreateVecs(raw_matrix, &near_nullspace_vector,
                                         nullptr),
                           "MatCreateVecs(near-nullspace)");
    try {
      meshaware::petsc_check(VecSet(near_nullspace_vector, 0.0),
                             "VecSet(near-nullspace)");
      std::vector<types::global_dof_index> dof_indices(dofs_per_polygon);
      std::vector<PetscInt> petsc_indices(dofs_per_polygon);
      std::vector<PetscScalar> petsc_values(dofs_per_polygon);
      for (const auto &polytope :
           agglomeration_handler->polytope_iterators()) {
        polytope->get_dof_indices(dof_indices);
        for (unsigned int index = 0; index < dofs_per_polygon; ++index) {
          petsc_indices[index] = static_cast<PetscInt>(dof_indices[index]);
          petsc_values[index] = constant_coefficients[index];
        }
        meshaware::petsc_check(
            VecSetValues(near_nullspace_vector,
                         static_cast<PetscInt>(dofs_per_polygon),
                         petsc_indices.data(), petsc_values.data(),
                         INSERT_VALUES),
            "VecSetValues(near-nullspace)");
      }
      meshaware::petsc_check(VecAssemblyBegin(near_nullspace_vector),
                             "VecAssemblyBegin(near-nullspace)");
      meshaware::petsc_check(VecAssemblyEnd(near_nullspace_vector),
                             "VecAssemblyEnd(near-nullspace)");
      PetscReal norm = 0.0;
      meshaware::petsc_check(VecNormalize(near_nullspace_vector, &norm),
                             "VecNormalize(near-nullspace)");
      if (!(norm > 0.0))
        throw std::runtime_error("constant near-nullspace has zero norm");

      MatNullSpace near_nullspace = nullptr;
      meshaware::petsc_check(
          MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_FALSE, 1,
                             &near_nullspace_vector, &near_nullspace),
          "MatNullSpaceCreate");
      const PetscErrorCode set_error =
          MatSetNearNullSpace(raw_matrix, near_nullspace);
      const PetscErrorCode destroy_error = MatNullSpaceDestroy(&near_nullspace);
      meshaware::petsc_check(set_error, "MatSetNearNullSpace");
      meshaware::petsc_check(destroy_error, "MatNullSpaceDestroy");
    } catch (...) {
      VecDestroy(&near_nullspace_vector);
      throw;
    }
    meshaware::petsc_check(VecDestroy(&near_nullspace_vector),
                           "VecDestroy(near-nullspace)");
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

  template <typename PolytopeIterator>
  double polytope_coefficient(const PolytopeIterator &polytope) const {
    const auto &background_cells = polytope->get_agglomerate();
    if (background_cells.empty())
      throw std::runtime_error("encountered an empty polytope");
    return coefficient(background_cells.front()->center());
  }

  void assemble_system() {
    const unsigned int dofs_per_cell = agglomeration_handler->n_dofs_per_cell();
    const bool assemble_trilinos =
        options.amg_backend ==
        meshaware::AmgBackend::polydeal_agglomeration;
    const double penalty_constant = 10.0 * (finite_element.get_degree() + 1.0) *
                                    (finite_element.get_degree() + 2.0);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> block_00(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> block_01(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> block_10(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> block_11(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> neighbor_dof_indices(dofs_per_cell);
    const ExactSolution exact_solution(options.pattern);

    for (const auto &polytope : agglomeration_handler->polytope_iterators()) {
      cell_matrix = 0.0;
      cell_rhs = 0.0;
      polytope->get_dof_indices(local_dof_indices);
      const auto &cell_values = agglomeration_handler->reinit(polytope);

      for (const unsigned int q : cell_values.quadrature_point_indices()) {
        const Point<2> &point = cell_values.get_quadrature_points()[q];
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

      for (unsigned int face = 0; face < polytope->n_faces(); ++face) {
        const double current_diameter = std::abs(polytope->diameter());
        if (polytope->at_boundary(face)) {
          const auto &face_values =
              agglomeration_handler->reinit(polytope, face);
          const auto &normals = face_values.get_normal_vectors();
          const double mu = polytope_coefficient(polytope);
          for (const unsigned int q : face_values.quadrature_point_indices()) {
            const Point<2> &point = face_values.get_quadrature_points()[q];
            const double penalty = penalty_constant * mu / current_diameter;
            const double boundary_value = exact_solution.value(point);
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const double normal_gradient_i =
                  face_values.shape_grad(i, q) * normals[q];
              cell_rhs(i) +=
                  (-mu * normal_gradient_i * boundary_value +
                   penalty * boundary_value * face_values.shape_value(i, q)) *
                  face_values.JxW(q);
              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const double normal_gradient_j =
                    face_values.shape_grad(j, q) * normals[q];
                cell_matrix(i, j) +=
                    (-mu * normal_gradient_i * face_values.shape_value(j, q) -
                     mu * normal_gradient_j * face_values.shape_value(i, q) +
                     penalty * face_values.shape_value(i, q) *
                         face_values.shape_value(j, q)) *
                    face_values.JxW(q);
              }
            }
          }
          continue;
        }

        const auto &neighbor = polytope->neighbor(face);
        if (polytope->index() >= neighbor->index())
          continue;

        const unsigned int neighbor_face =
            polytope->neighbor_of_agglomerated_neighbor(face);
        const auto &interface_values = agglomeration_handler->reinit_interface(
            polytope, neighbor, face, neighbor_face);
        const auto &values_0 = interface_values.first;
        const auto &values_1 = interface_values.second;
        const auto &normals = values_0.get_normal_vectors();
        const double neighbor_diameter = std::abs(neighbor->diameter());
        const double mu_0 = polytope_coefficient(polytope);
        const double mu_1 = polytope_coefficient(neighbor);
        neighbor->get_dof_indices(neighbor_dof_indices);
        block_00 = 0.0;
        block_01 = 0.0;
        block_10 = 0.0;
        block_11 = 0.0;

        for (const unsigned int q : values_0.quadrature_point_indices()) {
          const double penalty =
              penalty_constant *
              std::max(mu_0 / current_diameter, mu_1 / neighbor_diameter);
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              const double normal_grad_0_i =
                  values_0.shape_grad(i, q) * normals[q];
              const double normal_grad_0_j =
                  values_0.shape_grad(j, q) * normals[q];
              const double normal_grad_1_i =
                  values_1.shape_grad(i, q) * normals[q];
              const double normal_grad_1_j =
                  values_1.shape_grad(j, q) * normals[q];
              const double value_0_i = values_0.shape_value(i, q);
              const double value_0_j = values_0.shape_value(j, q);
              const double value_1_i = values_1.shape_value(i, q);
              const double value_1_j = values_1.shape_value(j, q);
              const double weight = values_0.JxW(q);

              block_00(i, j) += (-0.5 * mu_0 * normal_grad_0_i * value_0_j -
                                 0.5 * mu_0 * normal_grad_0_j * value_0_i +
                                 penalty * value_0_i * value_0_j) *
                                weight;
              block_01(i, j) += (0.5 * mu_0 * normal_grad_0_i * value_1_j -
                                 0.5 * mu_1 * normal_grad_1_j * value_0_i -
                                 penalty * value_0_i * value_1_j) *
                                weight;
              block_10(i, j) += (-0.5 * mu_1 * normal_grad_1_i * value_0_j +
                                 0.5 * mu_0 * normal_grad_0_j * value_1_i -
                                 penalty * value_1_i * value_0_j) *
                                weight;
              block_11(i, j) += (0.5 * mu_1 * normal_grad_1_i * value_1_j +
                                 0.5 * mu_1 * normal_grad_1_j * value_1_i +
                                 penalty * value_1_i * value_1_j) *
                                weight;
            }
          }
        }

        if (options.oracle)
          constraints.distribute_local_to_global(block_00, local_dof_indices,
                                                 system_matrix);
        constraints.distribute_local_to_global(block_00, local_dof_indices,
                                               petsc_matrix);
        if (assemble_trilinos)
          constraints.distribute_local_to_global(block_00, local_dof_indices,
                                                 trilinos_matrix);
        if (options.oracle)
          constraints.distribute_local_to_global(
              block_01, local_dof_indices, neighbor_dof_indices, system_matrix);
        constraints.distribute_local_to_global(
            block_01, local_dof_indices, neighbor_dof_indices, petsc_matrix);
        if (assemble_trilinos)
          constraints.distribute_local_to_global(
              block_01, local_dof_indices, neighbor_dof_indices,
              trilinos_matrix);
        if (options.oracle)
          constraints.distribute_local_to_global(
              block_10, neighbor_dof_indices, local_dof_indices, system_matrix);
        constraints.distribute_local_to_global(block_10, neighbor_dof_indices,
                                               local_dof_indices, petsc_matrix);
        if (assemble_trilinos)
          constraints.distribute_local_to_global(
              block_10, neighbor_dof_indices, local_dof_indices,
              trilinos_matrix);
        if (options.oracle)
          constraints.distribute_local_to_global(block_11, neighbor_dof_indices,
                                                 system_matrix);
        constraints.distribute_local_to_global(block_11, neighbor_dof_indices,
                                               petsc_matrix);
        if (assemble_trilinos)
          constraints.distribute_local_to_global(block_11,
                                                 neighbor_dof_indices,
                                                 trilinos_matrix);
      }

      if (options.oracle)
        constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                               local_dof_indices, system_matrix,
                                               right_hand_side);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                             local_dof_indices, petsc_matrix,
                                             petsc_right_hand_side);
      if (assemble_trilinos)
        constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices, trilinos_matrix,
            trilinos_right_hand_side);
    }

    petsc_matrix.compress(VectorOperation::add);
    petsc_right_hand_side.compress(VectorOperation::add);
    if (assemble_trilinos) {
      trilinos_matrix.compress(VectorOperation::add);
      trilinos_right_hand_side.compress(VectorOperation::add);
    }
  }

  std::uint64_t matrix_nonzeros() const {
    MatInfo information;
    meshaware::petsc_check(MatGetInfo(static_cast<Mat>(petsc_matrix),
                                      MAT_GLOBAL_SUM, &information),
                           "MatGetInfo");
    return static_cast<std::uint64_t>(information.nz_used);
  }

  std::string matrix_id() const {
    return meshaware::make_matrix_id(
        "poly", options.level, meshaware::to_string(options.pattern),
        options.epsilon, meshaware::to_string(options.high_region));
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
    record.mesh_family = "polygonal";
    record.level = options.level;
    record.h_nominal = std::pow(2.0, -int(options.level));
    record.h_max = h_max;
    record.pattern = meshaware::to_string(options.pattern);
    record.epsilon = options.epsilon;
    record.high_region = meshaware::to_string(options.high_region);
    record.theta = theta;
    record.amg_backend = meshaware::to_string(options.amg_backend);
    record.boomeramg_profile =
        meshaware::to_string(options.boomeramg_profile);
    record.amg_smoother = meshaware::to_string(options.amg_smoother);
    record.amg_relaxation_weight =
        options.amg_smoother == meshaware::AmgSmoother::damped_jacobi
            ? options.jacobi_damping
            : 1.0;
    record.repeat = repeat;
    record.cells = agglomeration_handler->n_agglomerates();
    record.background_cells = triangulation.n_active_cells();
    record.dofs = agglomeration_handler->n_dofs();
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

  double matrix_symmetry_defect() const {
    double maximum_defect = 0.0;
    double maximum_entry = 0.0;
    for (unsigned int row = 0; row < system_matrix.m(); ++row)
      for (auto entry = system_matrix.begin(row);
           entry != system_matrix.end(row); ++entry) {
        maximum_defect = std::max(
            maximum_defect,
            std::abs(entry->value() - system_matrix.el(entry->column(), row)));
        maximum_entry = std::max(maximum_entry, std::abs(entry->value()));
      }
    return maximum_entry == 0.0 ? 0.0 : maximum_defect / maximum_entry;
  }

  void solve_direct() {
    SparseDirectUMFPACK direct_solver;
    direct_solver.initialize(system_matrix);
    direct_solver.vmult(solution, right_hand_side);
  }

  double matrix_equivalence_defect() const {
    double maximum_defect = 0.0;
    const Mat raw_matrix = static_cast<Mat>(petsc_matrix);
    for (unsigned int row = 0; row < system_matrix.m(); ++row) {
      for (auto entry = system_matrix.begin(row);
           entry != system_matrix.end(row); ++entry) {
        const PetscInt petsc_row = row;
        const PetscInt petsc_column = entry->column();
        PetscScalar petsc_value = 0.0;
        meshaware::petsc_check(MatGetValues(raw_matrix, 1, &petsc_row, 1,
                                            &petsc_column, &petsc_value),
                               "MatGetValues");
        maximum_defect =
            std::max(maximum_defect,
                     std::abs(entry->value() - PetscRealPart(petsc_value)));
      }
    }
    return maximum_defect;
  }

  double rhs_equivalence_defect() const {
    double maximum_defect = 0.0;
    const Vec raw_rhs = static_cast<const Vec &>(petsc_right_hand_side);
    for (unsigned int index = 0; index < right_hand_side.size(); ++index) {
      const PetscInt petsc_index = index;
      PetscScalar petsc_value = 0.0;
      meshaware::petsc_check(
          VecGetValues(raw_rhs, 1, &petsc_index, &petsc_value),
          "VecGetValues(rhs)");
      maximum_defect =
          std::max(maximum_defect, std::abs(right_hand_side[index] -
                                            PetscRealPart(petsc_value)));
    }
    return maximum_defect;
  }

  meshaware::SolverMetrics solve_system(const double theta) {
    if (options.amg_backend ==
        meshaware::AmgBackend::polydeal_agglomeration)
      return solve_with_polydeal_multigrid();

    const meshaware::AmgSolverOptions solver_options{
        options.relative_tolerance, options.absolute_tolerance,
        options.maximum_iterations, theta, options.amg_smoother,
        options.jacobi_damping, options.boomeramg_profile};
    const meshaware::SolverMetrics metrics = meshaware::solve_with_boomer_amg(
        static_cast<Mat>(petsc_matrix),
        static_cast<const Vec &>(petsc_right_hand_side),
        static_cast<const Vec &>(petsc_solution), solver_options);

    const Vec raw_solution = static_cast<const Vec &>(petsc_solution);
    const PetscScalar *values = nullptr;
    meshaware::petsc_check(VecGetArrayRead(raw_solution, &values),
                           "VecGetArrayRead(solution)");
    for (unsigned int index = 0; index < petsc_solution_native.size(); ++index)
      petsc_solution_native[index] = PetscRealPart(values[index]);
    meshaware::petsc_check(VecRestoreArrayRead(raw_solution, &values),
                           "VecRestoreArrayRead(solution)");
    return metrics;
  }

  meshaware::SolverMetrics solve_with_polydeal_multigrid() {
    using MatrixType = TrilinosWrappers::SparseMatrix;
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const auto setup_start = Clock::now();
    const unsigned int n_levels = mg_agglomeration_handlers.size();
    const unsigned int max_level = n_levels - 1;
    const MPI_Comm communicator =
        agglomeration_handler->agglo_dh.get_mpi_communicator();

    std::vector<MatrixType> injection_matrices(n_levels - 1);
    for (unsigned int level = 0; level < max_level; ++level)
      fill_nested_injection_matrix(
          *mg_agglomeration_handlers[level],
          *mg_agglomeration_handlers[level + 1], mg_level_children[level],
          injection_matrices[level]);

    AmgProjector<2, MatrixType, double> projector(injection_matrices);
    MGLevelObject<std::unique_ptr<MatrixType>> level_matrices(0, max_level);
    level_matrices[max_level] = std::make_unique<MatrixType>();
    level_matrices[max_level]->copy_from(trilinos_matrix);
    projector.compute_level_matrices(level_matrices);

    mg::Matrix<VectorType> multigrid_matrix(level_matrices);
    std::unique_ptr<MGSmootherBase<VectorType>> multigrid_smoother;
    std::vector<VectorType> inverse_diagonals(n_levels);

    if (options.amg_smoother == meshaware::AmgSmoother::chebyshev) {
      using Chebyshev = PreconditionChebyshev<MatrixType, VectorType>;
      using ChebyshevSmoother =
          mg::SmootherRelaxation<Chebyshev, VectorType>;
      auto smoother = std::make_unique<ChebyshevSmoother>();
      MGLevelObject<typename Chebyshev::AdditionalData> smoother_data(
          0, max_level);
      for (unsigned int level = 0; level <= max_level; ++level) {
        const auto &matrix = *level_matrices[level];
        inverse_diagonals[level].reinit(
            matrix.locally_owned_range_indices(), communicator);
        for (unsigned int row = matrix.local_range().first;
             row < matrix.local_range().second; ++row) {
          const double diagonal = matrix.diag_element(row);
          if (std::abs(diagonal) <=
              100.0 * std::numeric_limits<double>::epsilon())
            throw std::runtime_error(
                "zero diagonal encountered in PolyDeal multigrid hierarchy");
          inverse_diagonals[level][row] = 1.0 / diagonal;
        }
        inverse_diagonals[level].compress(VectorOperation::insert);
        smoother_data[level].preconditioner =
            std::make_shared<DiagonalMatrix<VectorType>>(
                inverse_diagonals[level]);
        smoother_data[level].smoothing_range = 20.0;
        smoother_data[level].degree = 3;
        smoother_data[level].eig_cg_n_iterations =
            std::min<unsigned int>(20, matrix.m());
      }
      smoother->set_steps(5);
      smoother->initialize(level_matrices, smoother_data);
      multigrid_smoother = std::move(smoother);
    } else if (options.amg_smoother ==
               meshaware::AmgSmoother::damped_jacobi) {
      using Jacobi = TrilinosWrappers::PreconditionJacobi;
      using JacobiSmoother =
          MGSmootherPrecondition<MatrixType, Jacobi, VectorType>;
      auto smoother = std::make_unique<JacobiSmoother>(5);
      smoother->initialize(
          level_matrices,
          Jacobi::AdditionalData(options.jacobi_damping, 0.0, 1));
      multigrid_smoother = std::move(smoother);
    } else {
      using SymmetricGaussSeidel = TrilinosWrappers::PreconditionSSOR;
      using SymmetricGaussSeidelSmoother =
          MGSmootherPrecondition<MatrixType, SymmetricGaussSeidel,
                                 VectorType>;
      auto smoother = std::make_unique<SymmetricGaussSeidelSmoother>(5);
      smoother->initialize(
          level_matrices,
          SymmetricGaussSeidel::AdditionalData(1.0, 0.0, 0, 1));
      multigrid_smoother = std::move(smoother);
    }

    Utils::MGCoarseDirect<VectorType, MatrixType,
                          TrilinosWrappers::SolverDirect>
        coarse_solver(*level_matrices[0]);

    MGLevelObject<MatrixType *> level_transfers(0, max_level);
    for (unsigned int level = 0; level < max_level; ++level)
      level_transfers[level] = &injection_matrices[level];
    level_transfers[max_level] = nullptr;

    std::vector<DoFHandler<2> *> dof_handlers(n_levels);
    for (unsigned int level = 0; level <= max_level; ++level)
      dof_handlers[level] = &mg_agglomeration_handlers[level]->agglo_dh;
    MGTransferAgglomeration<2, VectorType> transfer(level_transfers,
                                                     dof_handlers);

    Multigrid<VectorType> multigrid(
        multigrid_matrix, coarse_solver, transfer, *multigrid_smoother,
        *multigrid_smoother, 0, numbers::invalid_unsigned_int,
        Multigrid<VectorType>::v_cycle);
    PreconditionMG<2, VectorType,
                   MGTransferAgglomeration<2, VectorType>>
        preconditioner(agglomeration_handler->agglo_dh, multigrid, transfer);
    const double setup_seconds = seconds_since(setup_start);

    trilinos_solution = 0.0;
    ReductionControl solver_control(
        options.maximum_iterations, options.absolute_tolerance,
        options.relative_tolerance);
    solver_control.enable_history_data();
    SolverCG<VectorType> solver(solver_control);
    const auto solve_start = Clock::now();
    solver.solve(trilinos_matrix, trilinos_solution,
                 trilinos_right_hand_side, preconditioner);
    const double solve_seconds = seconds_since(solve_start);

    const auto &history = solver_control.get_history_data();
    if (history.empty())
      throw std::runtime_error(
          "PolyDeal multigrid returned an empty residual history");
    const unsigned int iterations = solver_control.last_step();
    const double residual_initial = history.front();
    const double residual_final = history.back();
    const double convergence_factor =
        iterations == 0 || residual_initial == 0.0
            ? 0.0
            : std::pow(residual_final / residual_initial,
                       1.0 / static_cast<double>(iterations));

    double hierarchy_rows = 0.0;
    double hierarchy_nonzeros = 0.0;
    for (unsigned int level = 0; level <= max_level; ++level) {
      hierarchy_rows += static_cast<double>(level_matrices[level]->m());
      hierarchy_nonzeros += static_cast<double>(
          level_matrices[level]->n_nonzero_elements());
    }
    const double grid_complexity =
        hierarchy_rows / static_cast<double>(level_matrices[max_level]->m());
    const double operator_complexity =
        hierarchy_nonzeros /
        static_cast<double>(level_matrices[max_level]->n_nonzero_elements());

    for (unsigned int index = 0; index < petsc_solution_native.size();
         ++index)
      petsc_solution_native[index] = trilinos_solution[index];

    return {iterations,
            n_levels,
            residual_initial,
            residual_final,
            convergence_factor,
            setup_seconds,
            solve_seconds,
            grid_complexity,
            operator_complexity,
            KSP_CONVERGED_RTOL};
  }

  double solution_equivalence_defect() const {
    double maximum_defect = 0.0;
    for (unsigned int index = 0; index < solution.size(); ++index)
      maximum_defect =
          std::max(maximum_defect,
                   std::abs(solution[index] - petsc_solution_native[index]));
    return maximum_defect;
  }

  ErrorNorms compute_errors(const Vector<double> &evaluated_solution) const {
    const unsigned int dofs_per_cell = agglomeration_handler->n_dofs_per_cell();
    const double penalty_constant = 10.0 * (finite_element.get_degree() + 1.0) *
                                    (finite_element.get_degree() + 2.0);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> neighbor_dof_indices(dofs_per_cell);
    const ExactSolution exact_solution(options.pattern);
    double l2_squared = 0.0;
    double broken_h1_squared = 0.0;
    double energy_squared = 0.0;

    for (const auto &polytope : agglomeration_handler->polytope_iterators()) {
      polytope->get_dof_indices(local_dof_indices);
      const auto &cell_values = agglomeration_handler->reinit(polytope);
      for (const unsigned int q : cell_values.quadrature_point_indices()) {
        const Point<2> &point = cell_values.get_quadrature_points()[q];
        double numerical_value = 0.0;
        Tensor<1, 2> numerical_gradient;
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          numerical_value += evaluated_solution(local_dof_indices[i]) *
                             cell_values.shape_value(i, q);
          numerical_gradient += evaluated_solution(local_dof_indices[i]) *
                                cell_values.shape_grad(i, q);
        }

        const double value_error =
            numerical_value - exact_solution.value(point);
        const Tensor<1, 2> gradient_error =
            numerical_gradient - exact_solution.gradient(point);
        const double gradient_error_squared = gradient_error.norm_square();
        const double weight = cell_values.JxW(q);
        l2_squared += value_error * value_error * weight;
        broken_h1_squared += gradient_error_squared * weight;
        energy_squared += coefficient(point) * gradient_error_squared * weight;
      }

      const double current_diameter = std::abs(polytope->diameter());
      const double mu_0 = polytope_coefficient(polytope);
      for (unsigned int face = 0; face < polytope->n_faces(); ++face) {
        if (polytope->at_boundary(face)) {
          const auto &face_values =
              agglomeration_handler->reinit(polytope, face);
          const double penalty = penalty_constant * mu_0 / current_diameter;
          for (const unsigned int q : face_values.quadrature_point_indices()) {
            double numerical_value = 0.0;
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              numerical_value += evaluated_solution(local_dof_indices[i]) *
                                 face_values.shape_value(i, q);
            const double jump =
                numerical_value -
                exact_solution.value(face_values.get_quadrature_points()[q]);
            energy_squared += penalty * jump * jump * face_values.JxW(q);
          }
          continue;
        }

        const auto &neighbor = polytope->neighbor(face);
        if (polytope->index() >= neighbor->index())
          continue;

        const unsigned int neighbor_face =
            polytope->neighbor_of_agglomerated_neighbor(face);
        const auto &interface_values = agglomeration_handler->reinit_interface(
            polytope, neighbor, face, neighbor_face);
        const auto &values_0 = interface_values.first;
        const auto &values_1 = interface_values.second;
        neighbor->get_dof_indices(neighbor_dof_indices);
        const double mu_1 = polytope_coefficient(neighbor);
        const double penalty =
            penalty_constant * std::max(mu_0 / current_diameter,
                                        mu_1 / std::abs(neighbor->diameter()));

        for (const unsigned int q : values_0.quadrature_point_indices()) {
          double numerical_value_0 = 0.0;
          double numerical_value_1 = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            numerical_value_0 += evaluated_solution(local_dof_indices[i]) *
                                 values_0.shape_value(i, q);
            numerical_value_1 += evaluated_solution(neighbor_dof_indices[i]) *
                                 values_1.shape_value(i, q);
          }
          const double jump = numerical_value_0 - numerical_value_1;
          energy_squared += penalty * jump * jump * values_0.JxW(q);
        }
      }
    }

    return {std::sqrt(l2_squared), std::sqrt(broken_h1_squared),
            std::sqrt(energy_squared)};
  }

  Options options;
  Triangulation<2> triangulation;
  MappingQ1<2> mapping;
  FE_AggloDGP<2> finite_element;
  std::unique_ptr<GridTools::Cache<2>> cached_triangulation;
  std::vector<std::unique_ptr<AgglomerationHandler<2>>>
      mg_agglomeration_handlers;
  std::vector<std::vector<std::vector<unsigned int>>> mg_level_children;
  AgglomerationHandler<2> *agglomeration_handler = nullptr;
  AffineConstraints<double> constraints;
  DynamicSparsityPattern dynamic_sparsity;
  SparsityPattern sparsity;
  SparseMatrix<double> system_matrix;
  Vector<double> right_hand_side;
  Vector<double> solution;
  PETScWrappers::SparseMatrix petsc_matrix;
  PETScWrappers::MPI::Vector petsc_right_hand_side;
  PETScWrappers::MPI::Vector petsc_solution;
  TrilinosWrappers::SparseMatrix trilinos_matrix;
  LinearAlgebra::distributed::Vector<double> trilinos_right_hand_side;
  LinearAlgebra::distributed::Vector<double> trilinos_solution;
  Vector<double> petsc_solution_native;
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
    PolyDealExperiment(options).run();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
