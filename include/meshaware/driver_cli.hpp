#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace meshaware {

inline std::string require_option_value(int &index, const int argc,
                                        char **argv) {
  if (++index >= argc)
    throw std::invalid_argument(std::string("Missing value after ") +
                                argv[index - 1]);
  return argv[index];
}

inline std::vector<double> parse_float_list(const std::string_view raw,
                                            const std::string_view option) {
  std::vector<double> values;
  std::istringstream stream{std::string(raw)};
  std::string token;
  while (std::getline(stream, token, ','))
    if (!token.empty())
      values.push_back(std::stod(token));
  if (values.empty())
    throw std::invalid_argument(std::string(option) + " must not be empty");
  return values;
}

} // namespace meshaware
