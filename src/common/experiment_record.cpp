#include "meshaware/experiment_record.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace meshaware {
namespace {
std::string json_escape(const std::string_view value) {
  std::ostringstream escaped;
  for (const char character : value) {
    switch (character) {
    case '"':
      escaped << "\\\"";
      break;
    case '\\':
      escaped << "\\\\";
      break;
    case '\b':
      escaped << "\\b";
      break;
    case '\f':
      escaped << "\\f";
      break;
    case '\n':
      escaped << "\\n";
      break;
    case '\r':
      escaped << "\\r";
      break;
    case '\t':
      escaped << "\\t";
      break;
    default:
      escaped << character;
    }
  }
  return escaped.str();
}

void write_json(std::ostream &output, const ExperimentRecord &record) {
  output << std::setprecision(17) << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"sample_id\": \"" << json_escape(record.sample_id) << "\",\n"
         << "  \"matrix_id\": \"" << json_escape(record.matrix_id) << "\",\n"
         << "  \"problem\": \"" << json_escape(record.problem) << "\",\n"
         << "  \"mesh_family\": \"" << json_escape(record.mesh_family)
         << "\",\n"
         << "  \"level\": " << record.level << ",\n"
         << "  \"h_nominal\": " << record.h_nominal << ",\n"
         << "  \"h_max\": " << record.h_max << ",\n"
         << "  \"pattern\": \"" << json_escape(record.pattern) << "\",\n"
         << "  \"epsilon\": " << record.epsilon << ",\n"
         << "  \"high_region\": \"" << json_escape(record.high_region)
         << "\",\n"
         << "  \"theta\": " << record.theta << ",\n"
         << "  \"amg_backend\": \"" << json_escape(record.amg_backend)
         << "\",\n"
         << "  \"boomeramg_profile\": \""
         << json_escape(record.boomeramg_profile) << "\",\n"
         << "  \"amg_smoother\": \"" << json_escape(record.amg_smoother)
         << "\",\n"
         << "  \"amg_relaxation_weight\": "
         << record.amg_relaxation_weight << ",\n"
         << "  \"repeat\": " << record.repeat << ",\n"
         << "  \"cells\": " << record.cells << ",\n"
         << "  \"background_cells\": " << record.background_cells << ",\n"
         << "  \"dofs\": " << record.dofs << ",\n"
         << "  \"nnz\": " << record.nonzeros << ",\n"
         << "  \"cg_iterations\": " << record.cg_iterations << ",\n"
         << "  \"amg_levels\": " << record.amg_levels << ",\n"
         << "  \"ksp_converged_reason\": " << record.ksp_converged_reason
         << ",\n"
         << "  \"residual_initial\": " << record.residual_initial << ",\n"
         << "  \"residual_final\": " << record.residual_final << ",\n"
         << "  \"convergence_factor\": " << record.convergence_factor << ",\n"
         << "  \"grid_complexity\": " << record.grid_complexity << ",\n"
         << "  \"operator_complexity\": " << record.operator_complexity
         << ",\n"
         << "  \"l2_error\": " << record.l2_error << ",\n"
         << "  \"h1_seminorm_error\": " << record.h1_seminorm_error << ",\n"
         << "  \"energy_error\": " << record.energy_error << ",\n"
         << "  \"assembly_time_seconds\": " << record.assembly_time_seconds
         << ",\n"
         << "  \"amg_setup_time_seconds\": " << record.amg_setup_time_seconds
         << ",\n"
         << "  \"solve_time_seconds\": " << record.solve_time_seconds << ",\n"
         << "  \"matrix_format\": \"" << json_escape(record.matrix_format)
         << "\",\n"
         << "  \"matrix_path\": \"" << json_escape(record.matrix_path.string())
         << "\"\n"
         << "}\n";
}
} // namespace

void write_experiment_record(const ExperimentRecord &record,
                             const std::filesystem::path &destination) {
  if (destination.empty()) {
    write_json(std::cout, record);
    return;
  }

  if (destination.has_parent_path())
    std::filesystem::create_directories(destination.parent_path());
  std::filesystem::path temporary = destination;
  temporary += ".tmp";
  {
    std::ofstream output(temporary);
    if (!output)
      throw std::runtime_error("Cannot write " + temporary.string());
    write_json(output, record);
    output.close();
    if (!output)
      throw std::runtime_error("Cannot finish writing " + temporary.string());
  }
  std::filesystem::rename(temporary, destination);
}

} // namespace meshaware
