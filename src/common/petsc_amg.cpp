#include "meshaware/petsc_amg.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace meshaware {
namespace {
using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string petsc_real_option(const double value) {
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

constexpr const char *strong_threshold_option =
    "-meshaware_pc_hypre_boomeramg_strong_threshold";
constexpr const char *smoother_option =
    "-meshaware_pc_hypre_boomeramg_relax_type_all";
constexpr const char *relaxation_weight_option =
    "-meshaware_pc_hypre_boomeramg_relax_weight_all";
constexpr const char *coarsen_type_option =
    "-meshaware_pc_hypre_boomeramg_coarsen_type";
constexpr const char *interpolation_type_option =
    "-meshaware_pc_hypre_boomeramg_interp_type";
constexpr const char *nodal_coarsening_option =
    "-meshaware_pc_hypre_boomeramg_nodal_coarsen";
constexpr const char *vector_interpolation_option =
    "-meshaware_pc_hypre_boomeramg_vec_interp_variant";
constexpr const char *vector_interpolation_qmax_option =
    "-meshaware_pc_hypre_boomeramg_vec_interp_qmax";
} // namespace

AmgBackend parse_amg_backend(const std::string_view name) {
  if (name == "boomeramg")
    return AmgBackend::boomeramg;
  if (name == "polydeal-agglomeration")
    return AmgBackend::polydeal_agglomeration;
  throw std::invalid_argument(
      "AMG backend must be 'boomeramg' or 'polydeal-agglomeration'");
}

const char *to_string(const AmgBackend backend) {
  switch (backend) {
  case AmgBackend::boomeramg:
    return "boomeramg";
  case AmgBackend::polydeal_agglomeration:
    return "polydeal-agglomeration";
  }
  throw std::invalid_argument("unknown AMG backend");
}

AmgSmoother parse_amg_smoother(const std::string_view name) {
  if (name == "chebyshev")
    return AmgSmoother::chebyshev;
  if (name == "damped-jacobi")
    return AmgSmoother::damped_jacobi;
  if (name == "l1-symmetric-gauss-seidel")
    return AmgSmoother::l1_symmetric_gauss_seidel;
  if (name == "symmetric-gauss-seidel")
    return AmgSmoother::symmetric_gauss_seidel;
  throw std::invalid_argument(
      "AMG smoother must be 'chebyshev', 'damped-jacobi', "
      "'l1-symmetric-gauss-seidel', or 'symmetric-gauss-seidel'");
}

const char *to_string(const AmgSmoother smoother) {
  switch (smoother) {
  case AmgSmoother::chebyshev:
    return "chebyshev";
  case AmgSmoother::damped_jacobi:
    return "damped-jacobi";
  case AmgSmoother::l1_symmetric_gauss_seidel:
    return "l1-symmetric-gauss-seidel";
  case AmgSmoother::symmetric_gauss_seidel:
    return "symmetric-gauss-seidel";
  }
  throw std::invalid_argument("unknown AMG smoother");
}

BoomerAmgProfile parse_boomeramg_profile(const std::string_view name) {
  if (name == "default")
    return BoomerAmgProfile::default_options;
  if (name == "polygonal-nodal")
    return BoomerAmgProfile::polygonal_nodal;
  throw std::invalid_argument(
      "BoomerAMG profile must be 'default' or 'polygonal-nodal'");
}

const char *to_string(const BoomerAmgProfile profile) {
  switch (profile) {
  case BoomerAmgProfile::default_options:
    return "default";
  case BoomerAmgProfile::polygonal_nodal:
    return "polygonal-nodal";
  }
  throw std::invalid_argument("unknown BoomerAMG profile");
}

void petsc_check(const PetscErrorCode error, const char *operation) {
  if (error != PETSC_SUCCESS)
    throw std::runtime_error(std::string("PETSc call failed: ") + operation +
                             " (error " + std::to_string(error) + ")");
}

void write_petsc_matrix(const Mat matrix,
                        const std::filesystem::path &path) {
  if (path.has_parent_path())
    std::filesystem::create_directories(path.parent_path());
  PetscViewer viewer = nullptr;
  petsc_check(PetscViewerBinaryOpen(PETSC_COMM_SELF, path.string().c_str(),
                                    FILE_MODE_WRITE, &viewer),
              "PetscViewerBinaryOpen");
  const PetscErrorCode view_error = MatView(matrix, viewer);
  const PetscErrorCode destroy_error = PetscViewerDestroy(&viewer);
  petsc_check(view_error, "MatView");
  petsc_check(destroy_error, "PetscViewerDestroy");
}

SolverMetrics solve_with_boomer_amg(const Mat matrix, const Vec right_hand_side,
                                    const Vec solution,
                                    const AmgSolverOptions &options) {
  if (!(options.strong_threshold > 0.0 && options.strong_threshold < 1.0))
    throw std::invalid_argument("AMG strong threshold must lie in (0,1)");
  if (options.relative_tolerance <= 0.0 || options.absolute_tolerance < 0.0 ||
      options.maximum_iterations == 0)
    throw std::invalid_argument("invalid solver tolerances or iteration limit");
  if (!(options.jacobi_damping > 0.0 && options.jacobi_damping <= 1.0))
    throw std::invalid_argument("Jacobi damping must lie in (0,1]");

  KSP ksp = nullptr;
  PC preconditioner = nullptr;
  std::vector<const char *> configured_options;

  const auto clear_configured_options = [&configured_options]() {
    for (auto option = configured_options.rbegin();
         option != configured_options.rend(); ++option)
      PetscOptionsClearValue(nullptr, *option);
    configured_options.clear();
  };

  const auto set_option = [&configured_options](const char *name,
                                                 const char *value) {
    petsc_check(PetscOptionsSetValue(nullptr, name, value),
                "PetscOptionsSetValue");
    configured_options.push_back(name);
  };

  petsc_check(KSPCreate(PETSC_COMM_SELF, &ksp), "KSPCreate");
  try {
    petsc_check(KSPSetOptionsPrefix(ksp, "meshaware_"), "KSPSetOptionsPrefix");
    petsc_check(KSPSetOperators(ksp, matrix, matrix), "KSPSetOperators");
    petsc_check(KSPSetType(ksp, KSPCG), "KSPSetType");
    petsc_check(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED),
                "KSPSetNormType");
    petsc_check(KSPSetTolerances(ksp, options.relative_tolerance,
                                 options.absolute_tolerance, PETSC_DEFAULT,
                                 options.maximum_iterations),
                "KSPSetTolerances");
    petsc_check(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE),
                "KSPSetInitialGuessNonzero");
    petsc_check(KSPSetResidualHistory(
                    ksp, nullptr, options.maximum_iterations + 1, PETSC_TRUE),
                "KSPSetResidualHistory");
    petsc_check(KSPGetPC(ksp, &preconditioner), "KSPGetPC");
    petsc_check(PCSetType(preconditioner, PCHYPRE), "PCSetType");
    petsc_check(PCHYPRESetType(preconditioner, "boomeramg"), "PCHYPRESetType");

    if (options.profile == BoomerAmgProfile::polygonal_nodal) {
      PetscInt block_size = 1;
      petsc_check(MatGetBlockSize(matrix, &block_size), "MatGetBlockSize");
      if (block_size < 2)
        throw std::invalid_argument(
            "polygonal-nodal requires a matrix block size greater than one");
      MatNullSpace near_nullspace = nullptr;
      petsc_check(MatGetNearNullSpace(matrix, &near_nullspace),
                  "MatGetNearNullSpace");
      if (near_nullspace == nullptr)
        throw std::invalid_argument(
            "polygonal-nodal requires a matrix near-nullspace");
      set_option(coarsen_type_option, "HMIS");
      set_option(interpolation_type_option, "ext+i");
      set_option(nodal_coarsening_option, "1");
      set_option(vector_interpolation_option, "2");
      // There is one supplied near-nullspace vector, so at most one
      // additional Q entry per interpolation row is needed. Without this
      // cap HYPRE substantially increases operator complexity on the DG
      // hierarchy without improving the observed iteration count.
      set_option(vector_interpolation_qmax_option, "1");
    }

    const std::string threshold = petsc_real_option(options.strong_threshold);
    set_option(strong_threshold_option, threshold.c_str());

    const char *hypre_smoother = "symmetric-SOR/Jacobi";
    if (options.smoother == AmgSmoother::chebyshev)
      hypre_smoother = "Chebyshev";
    else if (options.smoother == AmgSmoother::damped_jacobi)
      hypre_smoother = "Jacobi";
    else if (options.smoother ==
             AmgSmoother::l1_symmetric_gauss_seidel)
      hypre_smoother = "l1scaled-SOR/Jacobi";
    set_option(smoother_option, hypre_smoother);

    const double relaxation_weight =
        options.smoother == AmgSmoother::damped_jacobi
            ? options.jacobi_damping
            : 1.0;
    const std::string weight = petsc_real_option(relaxation_weight);
    set_option(relaxation_weight_option, weight.c_str());

    petsc_check(KSPSetFromOptions(ksp), "KSPSetFromOptions");

    // Guarantee the same zero initial guess for every timing repetition.
    // This operation is intentionally outside the solve interval.
    petsc_check(VecSet(solution, 0.0), "VecSet(solution)");

    // The operator and all options are attached before this interval.
    const auto setup_start = Clock::now();
    petsc_check(KSPSetUp(ksp), "KSPSetUp");
    const double setup_seconds = seconds_since(setup_start);

    // KSPSetUp above prevents lazy AMG construction from entering this timer.
    const auto solve_start = Clock::now();
    petsc_check(KSPSolve(ksp, right_hand_side, solution), "KSPSolve");
    const double solve_seconds = seconds_since(solve_start);

    PetscInt iterations = 0;
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    const PetscReal *history = nullptr;
    PetscInt history_length = 0;
    petsc_check(KSPGetIterationNumber(ksp, &iterations),
                "KSPGetIterationNumber");
    petsc_check(KSPGetConvergedReason(ksp, &reason), "KSPGetConvergedReason");
    petsc_check(KSPGetResidualHistory(ksp, &history, &history_length),
                "KSPGetResidualHistory");

    if (reason < 0)
      throw std::runtime_error("CG failed with KSP reason " +
                               std::to_string(reason));
    if (history_length < 1)
      throw std::runtime_error("PETSc returned an empty residual history");

    const double residual_initial = history[0];
    const double residual_final = history[history_length - 1];
    const double convergence_factor =
        iterations == 0 || residual_initial == 0.0
            ? 0.0
            : std::pow(residual_final / residual_initial,
                       1.0 / static_cast<double>(iterations));

    // This public PETSc query reports the hierarchy depth, including the
    // finest level. It transfers ownership of the coarse matrix wrappers, so
    // use it only after all timed work and destroy those wrappers immediately.
    PetscInt amg_levels = 0;
    Mat *coarse_operators = nullptr;
    petsc_check(
        PCGetCoarseOperators(preconditioner, &amg_levels, &coarse_operators),
        "PCGetCoarseOperators");
    if (amg_levels < 1)
      throw std::runtime_error("BoomerAMG returned an empty hierarchy");
    PetscInt fine_rows = 0;
    PetscInt fine_columns = 0;
    MatInfo fine_information;
    petsc_check(MatGetSize(matrix, &fine_rows, &fine_columns), "MatGetSize");
    petsc_check(MatGetInfo(matrix, MAT_GLOBAL_SUM, &fine_information),
                "MatGetInfo");
    if (fine_rows < 1 || fine_columns != fine_rows ||
        fine_information.nz_used <= 0.0)
      throw std::runtime_error("invalid fine operator dimensions");
    double hierarchy_rows = static_cast<double>(fine_rows);
    double hierarchy_nonzeros = fine_information.nz_used;
    // PETSc fills one wrapper per coarse level. Its final allocation slot
    // represents the fine operator and is intentionally left unpopulated.
    for (PetscInt level = 0; level + 1 < amg_levels; ++level) {
      PetscInt rows = 0;
      PetscInt columns = 0;
      MatInfo information;
      petsc_check(MatGetSize(coarse_operators[level], &rows, &columns),
                  "MatGetSize(coarse operator)");
      petsc_check(MatGetInfo(coarse_operators[level], MAT_GLOBAL_SUM,
                             &information),
                  "MatGetInfo(coarse operator)");
      hierarchy_rows += static_cast<double>(rows);
      hierarchy_nonzeros += information.nz_used;
      petsc_check(MatDestroy(&coarse_operators[level]),
                  "MatDestroy(coarse operator)");
    }
    petsc_check(PetscFree(coarse_operators), "PetscFree(coarse operators)");
    const double grid_complexity =
        hierarchy_rows / static_cast<double>(fine_rows);
    const double operator_complexity =
        hierarchy_nonzeros / fine_information.nz_used;

    clear_configured_options();
    petsc_check(KSPDestroy(&ksp), "KSPDestroy");

    return {static_cast<unsigned int>(iterations),
            static_cast<unsigned int>(amg_levels),
            residual_initial,
            residual_final,
            convergence_factor,
            setup_seconds,
            solve_seconds,
            grid_complexity,
            operator_complexity,
            reason};
  } catch (...) {
    clear_configured_options();
    if (ksp != nullptr)
      KSPDestroy(&ksp);
    throw;
  }
}

} // namespace meshaware
