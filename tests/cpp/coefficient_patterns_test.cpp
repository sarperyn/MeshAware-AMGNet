#include "meshaware/coefficient_patterns.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
  using namespace meshaware;

  assert(parse_pattern("vertical_split") == Pattern::vertical_split);
  assert(exact_wavenumber(Pattern::checkerboard_2x2) == 1);
  assert(exact_wavenumber(Pattern::checkerboard_4x4) == 2);
  assert((tile_counts(Pattern::vertical_split) ==
          std::array<unsigned int, 2>{{2, 1}}));
  assert((tile_counts(Pattern::checkerboard_2x2) ==
          std::array<unsigned int, 2>{{2, 2}}));
  assert((tile_counts(Pattern::vertical_stripes_4) ==
          std::array<unsigned int, 2>{{4, 1}}));
  assert((tile_counts(Pattern::checkerboard_4x4) ==
          std::array<unsigned int, 2>{{4, 4}}));

  assert(is_white(Pattern::vertical_split, -0.5, 0.0));
  assert(!is_white(Pattern::vertical_split, 0.5, 0.0));

  assert(is_white(Pattern::checkerboard_2x2, -0.5, 0.5));
  assert(!is_white(Pattern::checkerboard_2x2, 0.5, 0.5));
  assert(!is_white(Pattern::checkerboard_2x2, -0.5, -0.5));
  assert(is_white(Pattern::checkerboard_2x2, 0.5, -0.5));

  assert(is_white(Pattern::vertical_stripes_4, -0.75, 0.0));
  assert(!is_white(Pattern::vertical_stripes_4, -0.25, 0.0));
  assert(is_white(Pattern::vertical_stripes_4, 0.25, 0.0));
  assert(!is_white(Pattern::vertical_stripes_4, 0.75, 0.0));

  assert(is_white(Pattern::checkerboard_4x4, -0.75, 0.75));
  assert(!is_white(Pattern::checkerboard_4x4, -0.25, 0.75));

  assert(diffusion_coefficient(Pattern::vertical_split, 2.0, -0.5, 0.0,
                               HighRegion::white) == 100.0);
  assert(diffusion_coefficient(Pattern::vertical_split, 2.0, -0.5, 0.0,
                               HighRegion::gray) == 1.0);
  assert(diffusion_coefficient(Pattern::checkerboard_4x4, 0.0, 0.1, 0.1) ==
         1.0);

  const double expected = 2.0 * std::acos(-1.0) * std::acos(-1.0);
  assert(std::abs(forcing_value(Pattern::vertical_split, 0.0, 0.0, 0.0) -
                  expected) < 1e-12);

  // The paper prescribes exact Dirichlet traces, not homogeneous values.
  assert(std::abs(exact_value(Pattern::vertical_split, -1.0, 0.0) + 1.0) <
         1e-12);

  std::cout << "coefficient pattern tests passed\n";
}
