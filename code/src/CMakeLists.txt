add_subdirectory(structs)
add_subdirectory(mp_raw)

# file(GLOB _HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
# file(GLOB _SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# file(GLOB _PERF_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/perf/*.cpp)

add_executable(iterators main.cpp)
target_include_directories(iterators PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(iterators structs mp_raw)

# add_executable(perf ${_HEADERS} ${_PERF_SOURCES})

set_target_properties(iterators PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set_target_properties(perf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
