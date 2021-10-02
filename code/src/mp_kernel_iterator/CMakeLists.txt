file(GLOB _HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(mp_kernel_iterator STATIC ${_HEADERS})
set_target_properties(mp_kernel_iterator PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(mp_kernel_iterator PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mp_kernel_iterator structs)
