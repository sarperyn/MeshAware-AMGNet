#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace meshaware {
enum class Pattern {
  vertical_split,
  checkerboard_2x2,
  vertical_stripes_4,
  checkerboard_4x4
};

enum class HighRegion { white, gray };

inline Pattern parse_pattern(const std::string_view name) {
  if (name == "vertical_split")
    return Pattern::vertical_split;
  if (name == "checkerboard_2x2")
    return Pattern::checkerboard_2x2;
  if (name == "vertical_stripes_4")
    return Pattern::vertical_stripes_4;
  if (name == "checkerboard_4x4")
    return Pattern::checkerboard_4x4;
  throw std::invalid_argument("Unknown diffusion pattern: " +
                              std::string(name));
}

inline const char *to_string(const Pattern pattern) {
  switch (pattern) {
  case Pattern::vertical_split:
    return "vertical_split";
  case Pattern::checkerboard_2x2:
    return "checkerboard_2x2";
  case Pattern::vertical_stripes_4:
    return "vertical_stripes_4";
  case Pattern::checkerboard_4x4:
    return "checkerboard_4x4";
  }
  throw std::logic_error("Unhandled diffusion pattern");
}

inline HighRegion parse_high_region(const std::string_view name) {
  if (name == "white")
    return HighRegion::white;
  if (name == "gray")
    return HighRegion::gray;
  throw std::invalid_argument("High region must be 'white' or 'gray'");
}

inline const char *to_string(const HighRegion region) {
  return region == HighRegion::white ? "white" : "gray";
}

inline unsigned int tile_index(const double coordinate,
                               const unsigned int tiles) {
  if (coordinate < -1.0 || coordinate > 1.0)
    throw std::out_of_range("Pattern coordinates must lie in [-1,1]");

  const double scaled = 0.5 * (coordinate + 1.0) * tiles;
  return std::min(static_cast<unsigned int>(std::floor(scaled)), tiles - 1);
}

// The colors match Figure 1: its upper-left tile is white.
inline bool is_white(const Pattern pattern, const double x, const double y) {
  switch (pattern) {
  case Pattern::vertical_split:
    return tile_index(x, 2) == 0;

  case Pattern::checkerboard_2x2:
    return (tile_index(x, 2) + tile_index(y, 2)) % 2 == 1;

  case Pattern::vertical_stripes_4:
    return tile_index(x, 4) % 2 == 0;

  case Pattern::checkerboard_4x4:
    return (tile_index(x, 4) + tile_index(y, 4)) % 2 == 1;
  }
  throw std::logic_error("Unhandled diffusion pattern");
}

inline double
diffusion_coefficient(const Pattern pattern, const double epsilon,
                      const double x, const double y,
                      const HighRegion high_region = HighRegion::white) {
  if (epsilon < 0.0)
    throw std::invalid_argument("epsilon must be non-negative");

  const bool point_is_high =
      is_white(pattern, x, y) == (high_region == HighRegion::white);
  return point_is_high ? std::pow(10.0, epsilon) : 1.0;
}

inline unsigned int exact_wavenumber(const Pattern pattern) {
  return pattern == Pattern::vertical_split ||
                 pattern == Pattern::checkerboard_2x2
             ? 1
             : 2;
}

inline std::array<unsigned int, 2> tile_counts(const Pattern pattern) {
  switch (pattern) {
  case Pattern::vertical_split:
    return {{2, 1}};
  case Pattern::checkerboard_2x2:
    return {{2, 2}};
  case Pattern::vertical_stripes_4:
    return {{4, 1}};
  case Pattern::checkerboard_4x4:
    return {{4, 4}};
  }
  throw std::logic_error("Unhandled diffusion pattern");
}

inline double exact_value(const Pattern pattern, const double x,
                          const double y) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double k = exact_wavenumber(pattern);
  return std::cos(k * pi * x) * std::cos(k * pi * y);
}

inline std::array<double, 2> exact_gradient(const Pattern pattern,
                                            const double x, const double y) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double kp = exact_wavenumber(pattern) * pi;
  return {{-kp * std::sin(kp * x) * std::cos(kp * y),
           -kp * std::cos(kp * x) * std::sin(kp * y)}};
}

// Inside each constant-coefficient tile:
// -div(mu grad u) = 2 (k pi)^2 mu u.
inline double forcing_value(const Pattern pattern, const double epsilon,
                            const double x, const double y,
                            const HighRegion high_region = HighRegion::white) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double kp = exact_wavenumber(pattern) * pi;
  return 2.0 * kp * kp *
         diffusion_coefficient(pattern, epsilon, x, y, high_region) *
         exact_value(pattern, x, y);
}
} // namespace meshaware
