#pragma once

#include <petscksp.h>

#include <filesystem>
#include <string_view>

namespace meshaware {

enum class AmgBackend {
  boomeramg,
  polydeal_agglomeration,
};

AmgBackend parse_amg_backend(std::string_view name);
const char *to_string(AmgBackend backend);

enum class AmgSmoother {
  chebyshev,
  damped_jacobi,
  l1_symmetric_gauss_seidel,
  symmetric_gauss_seidel,
};

AmgSmoother parse_amg_smoother(std::string_view name);
const char *to_string(AmgSmoother smoother);

enum class BoomerAmgProfile {
  default_options,
  polygonal_nodal,
};

BoomerAmgProfile parse_boomeramg_profile(std::string_view name);
const char *to_string(BoomerAmgProfile profile);

struct AmgSolverOptions {
  double relative_tolerance = 1e-8;
  double absolute_tolerance = 1e-50;
  unsigned int maximum_iterations = 10000;
  double strong_threshold = 0.24;
  AmgSmoother smoother = AmgSmoother::symmetric_gauss_seidel;
  double jacobi_damping = 2.0 / 3.0;
  BoomerAmgProfile profile = BoomerAmgProfile::default_options;
};

struct SolverMetrics {
  unsigned int iterations = 0;
  unsigned int amg_levels = 0;
  double residual_initial = 0.0;
  double residual_final = 0.0;
  double convergence_factor = 0.0;
  double setup_seconds = 0.0;
  double solve_seconds = 0.0;
  double grid_complexity = 0.0;
  double operator_complexity = 0.0;
  KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
};

void petsc_check(PetscErrorCode error, const char *operation);

void write_petsc_matrix(Mat matrix, const std::filesystem::path &path);

SolverMetrics solve_with_boomer_amg(Mat matrix, Vec right_hand_side,
                                    Vec solution,
                                    const AmgSolverOptions &options);

} // namespace meshaware
