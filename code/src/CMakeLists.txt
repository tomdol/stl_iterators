file(GLOB _HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB _SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB _PERF_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/perf/*.cpp)

message(${_SOURCES})
add_executable(iterators ${_HEADERS} ${_SOURCES})

add_executable(perf ${_HEADERS} ${_PERF_SOURCES})

set_target_properties(iterators PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(perf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
