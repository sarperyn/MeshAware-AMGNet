#include "meshaware/petsc_amg.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

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
} // namespace

AmgSmoother parse_amg_smoother(const std::string_view name) {
  if (name == "chebyshev")
    return AmgSmoother::chebyshev;
  if (name == "damped-jacobi")
    return AmgSmoother::damped_jacobi;
  if (name == "symmetric-gauss-seidel")
    return AmgSmoother::symmetric_gauss_seidel;
  throw std::invalid_argument(
      "AMG smoother must be 'chebyshev', 'damped-jacobi', or "
      "'symmetric-gauss-seidel'");
}

const char *to_string(const AmgSmoother smoother) {
  switch (smoother) {
  case AmgSmoother::chebyshev:
    return "chebyshev";
  case AmgSmoother::damped_jacobi:
    return "damped-jacobi";
  case AmgSmoother::symmetric_gauss_seidel:
    return "symmetric-gauss-seidel";
  }
  throw std::invalid_argument("unknown AMG smoother");
}

void petsc_check(const PetscErrorCode error, const char *operation) {
  if (error != PETSC_SUCCESS)
    throw std::runtime_error(std::string("PETSc call failed: ") + operation +
                             " (error " + std::to_string(error) + ")");
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
  bool threshold_option_is_set = false;
  bool smoother_option_is_set = false;
  bool relaxation_weight_option_is_set = false;

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

    const std::string threshold =
        petsc_real_option(options.strong_threshold);
    petsc_check(PetscOptionsSetValue(nullptr, strong_threshold_option,
                                     threshold.c_str()),
                "PetscOptionsSetValue(strong_threshold)");
    threshold_option_is_set = true;

    const char *hypre_smoother = "symmetric-SOR/Jacobi";
    if (options.smoother == AmgSmoother::chebyshev)
      hypre_smoother = "Chebyshev";
    else if (options.smoother == AmgSmoother::damped_jacobi)
      hypre_smoother = "Jacobi";
    petsc_check(PetscOptionsSetValue(nullptr, smoother_option, hypre_smoother),
                "PetscOptionsSetValue(smoother)");
    smoother_option_is_set = true;

    const double relaxation_weight =
        options.smoother == AmgSmoother::damped_jacobi
            ? options.jacobi_damping
            : 1.0;
    const std::string weight = petsc_real_option(relaxation_weight);
    petsc_check(PetscOptionsSetValue(nullptr, relaxation_weight_option,
                                     weight.c_str()),
                "PetscOptionsSetValue(relaxation_weight)");
    relaxation_weight_option_is_set = true;

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
    // PETSc fills one wrapper per coarse level. Its final allocation slot
    // represents the fine operator and is intentionally left unpopulated.
    for (PetscInt level = 0; level + 1 < amg_levels; ++level)
      petsc_check(MatDestroy(&coarse_operators[level]),
                  "MatDestroy(coarse operator)");
    petsc_check(PetscFree(coarse_operators), "PetscFree(coarse operators)");

    petsc_check(PetscOptionsClearValue(nullptr, strong_threshold_option),
                "PetscOptionsClearValue(strong_threshold)");
    threshold_option_is_set = false;
    petsc_check(PetscOptionsClearValue(nullptr, smoother_option),
                "PetscOptionsClearValue(smoother)");
    smoother_option_is_set = false;
    petsc_check(PetscOptionsClearValue(nullptr, relaxation_weight_option),
                "PetscOptionsClearValue(relaxation_weight)");
    relaxation_weight_option_is_set = false;
    petsc_check(KSPDestroy(&ksp), "KSPDestroy");

    return {static_cast<unsigned int>(iterations),
            static_cast<unsigned int>(amg_levels),
            residual_initial,
            residual_final,
            convergence_factor,
            setup_seconds,
            solve_seconds,
            reason};
  } catch (...) {
    if (threshold_option_is_set)
      PetscOptionsClearValue(nullptr, strong_threshold_option);
    if (smoother_option_is_set)
      PetscOptionsClearValue(nullptr, smoother_option);
    if (relaxation_weight_option_is_set)
      PetscOptionsClearValue(nullptr, relaxation_weight_option);
    if (ksp != nullptr)
      KSPDestroy(&ksp);
    throw;
  }
}

} // namespace meshaware
