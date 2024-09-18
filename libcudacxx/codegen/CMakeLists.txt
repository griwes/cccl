## Codegen adds the following targets for each codegen source:
## Build targets:
#    libcudacxx.codegen.${name}
#    libcudacxx.codegen.${name}.execute
#    libcudacxx.codegen.${name}.install
## Test targets:
#    libcudacxx.test.codegen.${name}.diff
## Current codegen sources:
#    atomic
#    float
#    float_test

include(${CMAKE_SOURCE_DIR}/cub/cmake/CPM.cmake)
CPMAddPackage("gh:fmtlib/fmt#11.0.1")

add_custom_target(libcudacxx.codegen)
add_custom_target(libcudacxx.codegen.execute)
add_custom_target(libcudacxx.codegen.install)

function(add_libcudacxx_codegen name destination)
    add_custom_target(libcudacxx.codegen.${name})
    add_dependencies(libcudacxx.codegen libcudacxx.codegen.${name})

    cmake_path(GET destination FILENAME filename)
    set(generated_output "${libcudacxx_BINARY_DIR}/codegen/${name}-${filename}")

    add_executable(
        codegen.${name}
        EXCLUDE_FROM_ALL
        ${name}.cpp
    )

    target_link_libraries(codegen.${name} PRIVATE fmt)

    set_property(TARGET codegen.${name} PROPERTY CXX_STANDARD 17)

    add_custom_target(
        libcudacxx.codegen.${name}.execute
        COMMAND codegen.${name} "${generated_output}"
        BYPRODUCTS "${generated_output}"
    )
    add_dependencies(libcudacxx.codegen.${name} libcudacxx.codegen.${name}.execute)
    add_dependencies(libcudacxx.codegen.execute libcudacxx.codegen.${name}.execute)

    add_custom_target(
        libcudacxx.codegen.${name}.install
        COMMAND ${CMAKE_COMMAND} -E copy "${generated_output}" "${destination}"
        DEPENDS libcudacxx.codegen.${name}
        BYPRODUCTS "${destination}"
    )
    add_dependencies(libcudacxx.codegen.${name}.install libcudacxx.codegen.${name}.execute)
    add_dependencies(libcudacxx.codegen.install libcudacxx.codegen.${name}.install)

    add_test(
        NAME libcudacxx.test.codegen.${name}.diff
        COMMAND ${CMAKE_COMMAND} -E compare_files "${destination}" "${generated_output}"
    )

    set_tests_properties(
        libcudacxx.test.codegen.${name}.diff
        PROPERTIES REQUIRED_FILES "${generated_output}"
    )
endfunction()

add_libcudacxx_codegen(atomic "${libcudacxx_SOURCE_DIR}/include/cuda/std/__atomic/functions/cuda_ptx_generated.h")
add_libcudacxx_codegen(float "${libcudacxx_SOURCE_DIR}/include/cuda/__float/generated.h")
add_libcudacxx_codegen(float_test "${libcudacxx_SOURCE_DIR}/test/libcudacxx/cuda/float/template.fp/testcases.h")
