#include "meshaware/coefficient_patterns.hpp"
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
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/numerics/vector_tools.h>

#include <agglomeration_handler.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
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

std::string slug_float(const double value) {
  std::ostringstream stream;
  stream << std::setprecision(10) << value;
  std::string result = stream.str();
  std::replace(result.begin(), result.end(), '.', 'p');
  std::replace(result.begin(), result.end(), '-', 'm');
  return result;
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
};

std::string require_value(int &index, const int argc, char **argv) {
  if (++index >= argc)
    throw std::invalid_argument(std::string("Missing value after ") +
                                argv[index - 1]);
  return argv[index];
}

std::vector<double> parse_float_list(const std::string &raw) {
  std::vector<double> values;
  std::istringstream stream(raw);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty())
      values.push_back(std::stod(token));
  }
  if (values.empty())
    throw std::invalid_argument("theta-values must not be empty");
  return values;
}

Options parse_options(const int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--mesh-family") {
      const std::string family = require_value(i, argc, argv);
      if (family != "polygonal")
        throw std::invalid_argument("PolyDeal mesh family must be 'polygonal'");
    } else if (argument == "--level")
      options.level = std::stoul(require_value(i, argc, argv));
    else if (argument == "--pattern")
      options.pattern = meshaware::parse_pattern(require_value(i, argc, argv));
    else if (argument == "--epsilon")
      options.epsilon = std::stod(require_value(i, argc, argv));
    else if (argument == "--high-region")
      options.high_region =
          meshaware::parse_high_region(require_value(i, argc, argv));
    else if (argument == "--theta")
      options.theta = std::stod(require_value(i, argc, argv));
    else if (argument == "--theta-values")
      options.theta_values = parse_float_list(require_value(i, argc, argv));
    else if (argument == "--rtol")
      options.relative_tolerance = std::stod(require_value(i, argc, argv));
    else if (argument == "--atol")
      options.absolute_tolerance = std::stod(require_value(i, argc, argv));
    else if (argument == "--max-iterations")
      options.maximum_iterations = std::stoul(require_value(i, argc, argv));
    else if (argument == "--repeat")
      options.repeat = std::stoul(require_value(i, argc, argv));
    else if (argument == "--repeats")
      options.repeats = std::stoul(require_value(i, argc, argv));
    else if (argument == "--warmup-runs")
      options.warmup_runs = std::stoul(require_value(i, argc, argv));
    else if (argument == "--record")
      options.record_path = require_value(i, argc, argv);
    else if (argument == "--record-dir")
      options.record_directory = require_value(i, argc, argv);
    else if (argument == "--matrix")
      options.matrix_path = require_value(i, argc, argv);
    else if (argument == "--skip-matrix-write")
      options.skip_matrix_write = true;
    else if (argument == "--skip-existing-records")
      options.skip_existing_records = true;
    else if (argument == "--oracle")
      options.oracle = true;
    else if (argument == "--help") {
      std::cout
          << "Usage: meshaware_diffusion_polydeal [options]\n"
          << "  --mesh-family polygonal\n"
          << "  --level N       nominal background h = 2^-N\n"
          << "  --pattern NAME  manufactured-solution pattern\n"
          << "  --epsilon E     coefficient contrast = 10^E\n"
          << "  --high-region NAME  white or gray\n"
          << "  --theta T       BoomerAMG strong threshold\n"
          << "  --theta-values CSV  batched strong-threshold grid\n"
          << "  --rtol T --atol T --max-iterations N\n"
          << "  --repeat N      timing repeat identifier\n"
          << "  --repeats N     repeats per theta in batch mode\n"
          << "  --warmup-runs N discarded warm-ups per theta\n"
          << "  --record PATH   JSON trial record\n"
          << "  --record-dir PATH  generated records for batch mode\n"
          << "  --matrix PATH   PETSc binary matrix/reference\n"
          << "  --skip-matrix-write\n"
          << "  --skip-existing-records\n"
          << "  --oracle        duplicate into native matrix and compare\n";
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
  if (options.repeats == 0)
    throw std::invalid_argument("repeats must be positive");
  if (!options.record_directory.empty() && !options.record_path.empty())
    throw std::invalid_argument("record and record-dir are mutually exclusive");
  if (options.record_directory.empty() && options.theta_values.size() != 1)
    throw std::invalid_argument("theta-values requires record-dir batch mode");
  if (options.oracle && !options.record_directory.empty())
    throw std::invalid_argument("oracle mode does not support batch execution");
  if (options.oracle && options.level > 6)
    throw std::invalid_argument("oracle mode is restricted to level <= 6");
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

    const std::uint64_t nonzeros = matrix_nonzeros();
    if (!options.matrix_path.empty() && !options.skip_matrix_write)
      write_matrix(options.matrix_path);

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
    const meshaware::SolverMetrics solver_metrics = solve_petsc(theta);
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
              << " cg_iterations=" << solver_metrics.iterations
              << " amg_levels=" << solver_metrics.amg_levels
              << " rho=" << solver_metrics.convergence_factor
              << " setup_s=" << solver_metrics.setup_seconds
              << " solve_s=" << solver_metrics.solve_seconds
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
        (void)solve_petsc(theta);

      for (const unsigned int repeat : pending_repeats) {
        const meshaware::SolverMetrics solver_metrics = solve_petsc(theta);
        const ErrorNorms errors = compute_errors(petsc_solution_native);
        write_record(theta, repeat, batch_record_path(theta, repeat),
                     assembly_seconds, solver_metrics, errors, nonzeros);
        std::cout << "polydeal"
                  << " polygons=" << agglomeration_handler->n_agglomerates()
                  << " dofs=" << agglomeration_handler->n_dofs()
                  << " theta=" << theta << " repeat=" << repeat
                  << " iterations=" << solver_metrics.iterations
                  << " amg_levels=" << solver_metrics.amg_levels
                  << " rho=" << solver_metrics.convergence_factor
                  << " setup_s=" << solver_metrics.setup_seconds
                  << " solve_s=" << solver_metrics.solve_seconds << '\n';
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
    agglomeration_handler =
        std::make_unique<AgglomerationHandler<2>>(*cached_triangulation);

    define_tile_constrained_agglomerates(subdivisions);
  }

  void define_tile_constrained_agglomerates(const unsigned int subdivisions) {
    using CellIterator = Triangulation<2>::active_cell_iterator;
    using AgglomerateKey = std::tuple<unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int>;
    std::map<AgglomerateKey, std::vector<CellIterator>> groups;

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
      groups[{tile_x, tile_y, block_x, block_y, first_polyomino ? 0u : 1u}]
          .push_back(cell);
    }

    for (const auto &[key, cells] : groups) {
      (void)key;
      if (cells.empty())
        throw std::runtime_error("constructed an empty agglomerate");
      agglomeration_handler->define_agglomerate(cells);
    }
  }

  void setup_system() {
    agglomeration_handler->distribute_agglomerated_dofs(finite_element);
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
        if (options.oracle)
          constraints.distribute_local_to_global(
              block_01, local_dof_indices, neighbor_dof_indices, system_matrix);
        constraints.distribute_local_to_global(
            block_01, local_dof_indices, neighbor_dof_indices, petsc_matrix);
        if (options.oracle)
          constraints.distribute_local_to_global(
              block_10, neighbor_dof_indices, local_dof_indices, system_matrix);
        constraints.distribute_local_to_global(block_10, neighbor_dof_indices,
                                               local_dof_indices, petsc_matrix);
        if (options.oracle)
          constraints.distribute_local_to_global(block_11, neighbor_dof_indices,
                                                 system_matrix);
        constraints.distribute_local_to_global(block_11, neighbor_dof_indices,
                                               petsc_matrix);
      }

      if (options.oracle)
        constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                               local_dof_indices, system_matrix,
                                               right_hand_side);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                             local_dof_indices, petsc_matrix,
                                             petsc_right_hand_side);
    }

    petsc_matrix.compress(VectorOperation::add);
    petsc_right_hand_side.compress(VectorOperation::add);
  }

  std::uint64_t matrix_nonzeros() const {
    MatInfo information;
    meshaware::petsc_check(MatGetInfo(static_cast<Mat>(petsc_matrix),
                                      MAT_GLOBAL_SUM, &information),
                           "MatGetInfo");
    return static_cast<std::uint64_t>(information.nz_used);
  }

  void write_matrix(const std::filesystem::path &path) const {
    if (path.has_parent_path())
      std::filesystem::create_directories(path.parent_path());
    PetscViewer viewer = nullptr;
    meshaware::petsc_check(PetscViewerBinaryOpen(PETSC_COMM_SELF,
                                                 path.string().c_str(),
                                                 FILE_MODE_WRITE, &viewer),
                           "PetscViewerBinaryOpen");
    const PetscErrorCode view_error =
        MatView(static_cast<Mat>(petsc_matrix), viewer);
    const PetscErrorCode destroy_error = PetscViewerDestroy(&viewer);
    meshaware::petsc_check(view_error, "MatView");
    meshaware::petsc_check(destroy_error, "PetscViewerDestroy");
  }

  std::string matrix_id() const {
    std::ostringstream stream;
    stream << "poly_l" << options.level << '_'
           << meshaware::to_string(options.pattern) << "_e"
           << slug_float(options.epsilon) << "_high_"
           << meshaware::to_string(options.high_region);
    return stream.str();
  }

  std::filesystem::path batch_record_path(const double theta,
                                          const unsigned int repeat) const {
    const std::string sample_id = matrix_id() + "_theta_" + slug_float(theta) +
                                  "_repeat_" + std::to_string(repeat);
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
    record.sample_id = id + "_theta_" + slug_float(theta) + "_repeat_" +
                       std::to_string(repeat);
    record.matrix_id = id;
    record.mesh_family = "polygonal";
    record.level = options.level;
    record.h_nominal = std::pow(2.0, -int(options.level));
    record.h_max = h_max;
    record.pattern = meshaware::to_string(options.pattern);
    record.epsilon = options.epsilon;
    record.high_region = meshaware::to_string(options.high_region);
    record.theta = theta;
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

  meshaware::SolverMetrics solve_petsc(const double theta) {
    const meshaware::AmgSolverOptions solver_options{
        options.relative_tolerance, options.absolute_tolerance,
        options.maximum_iterations, theta};
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
  std::unique_ptr<AgglomerationHandler<2>> agglomeration_handler;
  AffineConstraints<double> constraints;
  DynamicSparsityPattern dynamic_sparsity;
  SparsityPattern sparsity;
  SparseMatrix<double> system_matrix;
  Vector<double> right_hand_side;
  Vector<double> solution;
  PETScWrappers::SparseMatrix petsc_matrix;
  PETScWrappers::MPI::Vector petsc_right_hand_side;
  PETScWrappers::MPI::Vector petsc_solution;
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
