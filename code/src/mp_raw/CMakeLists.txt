file(GLOB _HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(mp_raw STATIC ${_HEADERS})
set_target_properties(mp_raw PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(mp_raw PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mp_raw structs)
