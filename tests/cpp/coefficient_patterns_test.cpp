#include "meshaware/coefficient_patterns.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace meshaware;
  const auto check = [](const bool condition) {
    if (!condition)
      throw std::runtime_error("coefficient pattern check failed");
  };

  check(parse_pattern("vertical_split") == Pattern::vertical_split);
  check(exact_wavenumber(Pattern::checkerboard_2x2) == 1);
  check(exact_wavenumber(Pattern::checkerboard_4x4) == 2);
  check((tile_counts(Pattern::vertical_split) ==
         std::array<unsigned int, 2>{{2, 1}}));
  check((tile_counts(Pattern::checkerboard_2x2) ==
         std::array<unsigned int, 2>{{2, 2}}));
  check((tile_counts(Pattern::vertical_stripes_4) ==
         std::array<unsigned int, 2>{{4, 1}}));
  check((tile_counts(Pattern::checkerboard_4x4) ==
         std::array<unsigned int, 2>{{4, 4}}));

  check(is_white(Pattern::vertical_split, -0.5, 0.0));
  check(!is_white(Pattern::vertical_split, 0.5, 0.0));

  check(is_white(Pattern::checkerboard_2x2, -0.5, 0.5));
  check(!is_white(Pattern::checkerboard_2x2, 0.5, 0.5));
  check(!is_white(Pattern::checkerboard_2x2, -0.5, -0.5));
  check(is_white(Pattern::checkerboard_2x2, 0.5, -0.5));

  check(is_white(Pattern::vertical_stripes_4, -0.75, 0.0));
  check(!is_white(Pattern::vertical_stripes_4, -0.25, 0.0));
  check(is_white(Pattern::vertical_stripes_4, 0.25, 0.0));
  check(!is_white(Pattern::vertical_stripes_4, 0.75, 0.0));

  check(is_white(Pattern::checkerboard_4x4, -0.75, 0.75));
  check(!is_white(Pattern::checkerboard_4x4, -0.25, 0.75));

  check(diffusion_coefficient(Pattern::vertical_split, 2.0, -0.5, 0.0,
                              HighRegion::white) == 100.0);
  check(diffusion_coefficient(Pattern::vertical_split, 2.0, -0.5, 0.0,
                              HighRegion::gray) == 1.0);
  check(diffusion_coefficient(Pattern::checkerboard_4x4, 0.0, 0.1, 0.1) ==
        1.0);

  const double expected = 2.0 * std::acos(-1.0) * std::acos(-1.0);
  check(std::abs(forcing_value(Pattern::vertical_split, 0.0, 0.0, 0.0) -
                 expected) < 1e-12);

  // The paper prescribes exact Dirichlet traces, not homogeneous values.
  check(std::abs(exact_value(Pattern::vertical_split, -1.0, 0.0) + 1.0) <
        1e-12);

  std::cout << "coefficient pattern tests passed\n";
}
