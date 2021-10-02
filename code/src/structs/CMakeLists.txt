file(GLOB _HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(structs STATIC ${_HEADERS})
set_target_properties(structs PROPERTIES LINKER_LANGUAGE CXX)
