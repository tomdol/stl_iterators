add_subdirectory(structs)
add_subdirectory(mp_raw)
add_subdirectory(mp_kernel_iterator)

# file(GLOB _PERF_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/perf/*.cpp)

add_executable(iterators main.cpp)
target_include_directories(iterators PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(iterators structs mp_raw mp_kernel_iterator)

add_executable(perf perf.cpp)
target_include_directories(perf PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(perf structs mp_raw mp_kernel_iterator)

set_target_properties(iterators PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(perf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
