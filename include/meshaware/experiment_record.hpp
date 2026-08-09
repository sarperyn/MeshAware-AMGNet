#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace meshaware {

struct ExperimentRecord {
  std::string sample_id;
  std::string matrix_id;
  std::string problem = "heterogeneous_diffusion";
  std::string mesh_family;
  unsigned int level = 0;
  double h_nominal = 0.0;
  double h_max = 0.0;
  std::string pattern;
  double epsilon = 0.0;
  std::string high_region;
  double theta = 0.0;
  std::string amg_backend = "boomeramg";
  std::string boomeramg_profile = "default";
  std::string amg_smoother = "symmetric-gauss-seidel";
  double amg_relaxation_weight = 1.0;
  unsigned int repeat = 0;
  std::uint64_t cells = 0;
  std::uint64_t background_cells = 0;
  std::uint64_t dofs = 0;
  std::uint64_t nonzeros = 0;
  unsigned int cg_iterations = 0;
  unsigned int amg_levels = 0;
  int ksp_converged_reason = 0;
  double residual_initial = 0.0;
  double residual_final = 0.0;
  double convergence_factor = 0.0;
  double grid_complexity = 0.0;
  double operator_complexity = 0.0;
  double l2_error = 0.0;
  double h1_seminorm_error = 0.0;
  double energy_error = 0.0;
  double assembly_time_seconds = 0.0;
  double amg_setup_time_seconds = 0.0;
  double solve_time_seconds = 0.0;
  std::string matrix_format = "petsc_binary";
  std::filesystem::path matrix_path;
};

std::string make_matrix_id(std::string_view prefix, unsigned int level,
                           std::string_view pattern, double epsilon,
                           std::string_view high_region);

std::string make_sample_id(std::string_view matrix_id, double theta,
                           unsigned int repeat);

void write_experiment_record(const ExperimentRecord &record,
                             const std::filesystem::path &destination);

} // namespace meshaware
