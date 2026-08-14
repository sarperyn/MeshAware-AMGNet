cmake_minimum_required(VERSION 3.20)

project(AMGThetaNet VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

add_library(meshaware_core INTERFACE)
target_include_directories(
  meshaware_core
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(meshaware_core INTERFACE cxx_std_17)

if(BUILD_TESTING)
  add_executable(coefficient_patterns_test tests/cpp/coefficient_patterns_test.cpp)
  target_link_libraries(coefficient_patterns_test PRIVATE meshaware_core)
  add_test(NAME coefficient_patterns COMMAND coefficient_patterns_test)
endif()

option(MESHAWARE_BUILD_DEALII "Build deal.II finite-element drivers" ON)
option(MESHAWARE_BUILD_POLYDEAL "Build the PolyDeal finite-element driver" ON)
set(POLYDEAL_ROOT "" CACHE PATH "PolyDeal source tree")

if(MESHAWARE_BUILD_DEALII)
  find_package(deal.II 9.7 QUIET
    HINTS
      ${deal.II_DIR}
      ${DEAL_II_DIR}
      ../deal.II
      ../../deal.II
  )

  if(deal.II_FOUND)
    if(NOT DEAL_II_WITH_PETSC)
      message(FATAL_ERROR "deal.II must be configured with PETSc support")
    endif()

    deal_ii_initialize_cached_variables()
    add_library(
      meshaware_petsc
      src/common/petsc_amg.cpp
      src/common/experiment_record.cpp
    )
    target_link_libraries(meshaware_petsc meshaware_core)
    deal_ii_setup_target(meshaware_petsc)

    add_executable(meshaware_diffusion_dealii src/dealii/diffusion_dealii.cpp)
    # deal_ii_setup_target() uses CMake's plain target_link_libraries
    # signature, so this target must use the same signature consistently.
    target_link_libraries(
      meshaware_diffusion_dealii
      meshaware_core
      meshaware_petsc
    )
    deal_ii_setup_target(meshaware_diffusion_dealii)

    add_executable(
      meshaware_diffusion_simplex_dg
      src/dealii/diffusion_simplex_dg.cpp
    )
    target_link_libraries(
      meshaware_diffusion_simplex_dg
      meshaware_core
      meshaware_petsc
    )
    deal_ii_setup_target(meshaware_diffusion_simplex_dg)

    if(MESHAWARE_BUILD_POLYDEAL)
      find_path(
        POLYDEAL_INCLUDE_DIR
        NAMES agglomeration_handler.h
        HINTS
          ${POLYDEAL_ROOT}/include
      )
      find_library(
        POLYDEAL_LIBRARY
        NAMES polydeal
        HINTS
          ${POLYDEAL_ROOT}/build-unified/source
          ${POLYDEAL_ROOT}/build/source
      )

      if(POLYDEAL_INCLUDE_DIR AND POLYDEAL_LIBRARY)
        if(NOT DEAL_II_WITH_TRILINOS)
          message(FATAL_ERROR
            "The PolyDeal agglomeration multigrid backend requires deal.II "
            "with Trilinos support.")
        endif()
        add_executable(
          meshaware_diffusion_polydeal
          src/polydeal/diffusion_polydeal.cpp
        )
        target_include_directories(
          meshaware_diffusion_polydeal
          PRIVATE ${POLYDEAL_INCLUDE_DIR}
        )
        target_link_libraries(
          meshaware_diffusion_polydeal
          meshaware_core
          meshaware_petsc
          ${POLYDEAL_LIBRARY}
        )
        deal_ii_setup_target(meshaware_diffusion_polydeal)

        if(BUILD_TESTING)
          add_test(
            NAME polydeal_production_smoke
            COMMAND meshaware_diffusion_polydeal
              --mesh-family polygonal
              --level 2
              --epsilon 2
              --pattern checkerboard_4x4
              --theta 0.24
          )
          add_test(
            NAME polydeal_native_oracle
            COMMAND meshaware_diffusion_polydeal
              --level 1
              --epsilon 0
              --pattern vertical_split
              --oracle
          )
          add_test(
            NAME polydeal_assemble_only
            COMMAND meshaware_diffusion_polydeal
              --mesh-family polygonal
              --level 2
              --epsilon 0
              --pattern vertical_split
              --matrix
                ${CMAKE_CURRENT_BINARY_DIR}/test_outputs/polydeal_assemble_only.petsc
              --assemble-only
          )
          add_test(
            NAME polydeal_chebyshev_smoke
            COMMAND meshaware_diffusion_polydeal
              --mesh-family polygonal
              --level 2
              --epsilon 2
              --pattern checkerboard_4x4
              --theta 0.24
              --amg-smoother chebyshev
          )
          add_test(
            NAME polydeal_boomeramg_nodal_smoke
            COMMAND meshaware_diffusion_polydeal
              --mesh-family polygonal
              --level 2
              --epsilon 7
              --pattern checkerboard_4x4
              --theta 0.24
              --amg-backend boomeramg
              --boomeramg-profile polygonal-nodal
              --amg-smoother symmetric-gauss-seidel
          )
          add_test(
            NAME polydeal_agglomeration_multigrid_smoke
            COMMAND meshaware_diffusion_polydeal
              --mesh-family polygonal
              --level 2
              --epsilon 2
              --pattern checkerboard_4x4
              --theta 0.24
              --amg-backend polydeal-agglomeration
              --amg-smoother chebyshev
              --oracle
          )
          add_test(
            NAME polydeal_agglomeration_damped_jacobi_smoke
            COMMAND meshaware_diffusion_polydeal
              --mesh-family polygonal
              --level 2
              --epsilon 2
              --pattern checkerboard_4x4
              --theta 0.24
              --amg-backend polydeal-agglomeration
              --amg-smoother damped-jacobi
              --jacobi-damping 0.6666666666666666
              --oracle
          )
          add_test(
            NAME polydeal_agglomeration_symmetric_gauss_seidel_smoke
            COMMAND meshaware_diffusion_polydeal
              --mesh-family polygonal
              --level 2
              --epsilon 2
              --pattern checkerboard_4x4
              --theta 0.24
              --amg-backend polydeal-agglomeration
              --amg-smoother symmetric-gauss-seidel
              --oracle
          )
          set_tests_properties(
            polydeal_production_smoke
            polydeal_native_oracle
            polydeal_assemble_only
            polydeal_chebyshev_smoke
            polydeal_boomeramg_nodal_smoke
            polydeal_agglomeration_multigrid_smoke
            polydeal_agglomeration_damped_jacobi_smoke
            polydeal_agglomeration_symmetric_gauss_seidel_smoke
            PROPERTIES ENVIRONMENT "OMPI_MCA_btl=self"
          )
          foreach(_pattern
              vertical_split
              checkerboard_2x2
              vertical_stripes_4
              checkerboard_4x4)
            add_test(
              NAME polydeal_heterogeneous_${_pattern}
              COMMAND meshaware_diffusion_polydeal
                --level 2
                --epsilon 2
                --pattern ${_pattern}
                --high-region white
                --theta 0.24
                --oracle
            )
            set_tests_properties(
              polydeal_heterogeneous_${_pattern}
              PROPERTIES ENVIRONMENT "OMPI_MCA_btl=self"
            )
          endforeach()
        endif()
      else()
        message(STATUS
          "PolyDeal was not found: skipping meshaware_diffusion_polydeal. "
          "Set POLYDEAL_ROOT to its source tree after building PolyDeal.")
      endif()
    endif()

    if(BUILD_TESTING)
      add_test(
        NAME dealii_quadrilateral_smoke
        COMMAND meshaware_diffusion_dealii
          --mesh-family quadrilateral
          --level 1
          --epsilon 0
          --pattern vertical_split
          --theta 0.24
      )
      add_test(
        NAME dealii_simplex_smoke
        COMMAND meshaware_diffusion_dealii
          --mesh-family simplex
          --level 1
          --epsilon 0
          --pattern vertical_split
          --theta 0.24
      )
      add_test(
        NAME dealii_damped_jacobi_smoke
        COMMAND meshaware_diffusion_dealii
          --mesh-family quadrilateral
          --level 1
          --epsilon 0
          --pattern vertical_split
          --theta 0.24
          --amg-smoother damped-jacobi
          --jacobi-damping 0.6666666666666666
      )
      add_test(
        NAME dealii_simplex_dg_smoke
        COMMAND meshaware_diffusion_simplex_dg
          --mesh-family simplex-dg
          --level 1
          --epsilon 0
          --pattern vertical_split
          --theta 0.24
          --verify
      )
      add_test(
        NAME dealii_simplex_dg_heterogeneous
        COMMAND meshaware_diffusion_simplex_dg
          --mesh-family simplex-dg
          --level 1
          --epsilon 9.5
          --pattern checkerboard_4x4
          --theta 0.24
          --verify
      )
      add_test(
        NAME dealii_assemble_only
        COMMAND meshaware_diffusion_dealii
          --mesh-family simplex
          --level 1
          --epsilon 0
          --pattern vertical_split
          --matrix
            ${CMAKE_CURRENT_BINARY_DIR}/test_outputs/dealii_assemble_only.petsc
          --assemble-only
      )
      set_tests_properties(
        dealii_quadrilateral_smoke
        dealii_simplex_smoke
        dealii_damped_jacobi_smoke
        dealii_simplex_dg_smoke
        dealii_simplex_dg_heterogeneous
        dealii_assemble_only
        PROPERTIES ENVIRONMENT "OMPI_MCA_btl=self"
      )
    endif()
  else()
    message(STATUS
      "deal.II >= 9.7 was not found: skipping meshaware_diffusion_dealii. "
      "Set DEAL_II_DIR to enable it.")
  endif()
endif()
