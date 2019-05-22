file(GLOB_RECURSE tools_srcs ${PROJECT_SOURCE_DIR}/tools/*.cpp) 

foreach(source_file ${tools_srcs})
  # get file name
  get_filename_component(name ${source_file} NAME_WE)

  # get folder name
  get_filename_component(path ${source_file} PATH)
  get_filename_component(folder ${path} NAME_WE)
  
  add_executable(${name} ${source_file})

  target_link_libraries(${name} boost_regex boost_program_options pthread)
  target_link_libraries(${name} admatch)
  target_link_libraries(${name} cuda)

  # set back RUNTIME_OUTPUT_DIRECTORY
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${folder}")

  # install
  install(TARGETS ${name} 
          RUNTIME DESTINATION bin 
          DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endforeach()
