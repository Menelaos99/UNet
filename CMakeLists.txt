if(BUILD_TESTING)
  add_subdirectory(texUnet)
endif()

install(FILES "tumletter.cls" DESTINATION ${LATEX_PACKAGE_DIR})
