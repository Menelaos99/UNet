if(BUILD_TESTING)
  add_subdirectory(tests_tumletter)
endif()

install(FILES "tumletter.cls" DESTINATION ${LATEX_PACKAGE_DIR})
