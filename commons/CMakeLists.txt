if(BUILD_TESTING)
  add_subdirectory(tests_tumauthblk)
  add_subdirectory(tests_tumboxes)
  add_subdirectory(tests_tumcaptions)
  add_subdirectory(tests_tumcolors)
  add_subdirectory(tests_tumfonts)
  add_subdirectory(tests_tumlang)
  add_subdirectory(tests_tumlogo)
  add_subdirectory(tests_tummath)
  add_subdirectory(tests_tumpage)

  add_custom_target(
    tests_commons
    DEPENDS tests_tumauthblk
            tests_tumboxes
            tests_tumcaptions
            tests_tumcolors
            tests_tumfonts
            tests_tumlang
            tests_tumlogo
            tests_tummath
            tests_tumpage
  )
endif()

set(INSTALL_FILES
    "tumauthblk.sty"
    "tumbase.sty"
    "tumboxes.sty"
    "tumcaptions.sty"
    "tumcolors.sty"
    "tumfonts.sty"
    "tumlang.sty"
    "tumlogo.sty"
    "tummath.sty"
    "tumpage.sty"
    "tumuser.sty"
)
install(FILES ${INSTALL_FILES} DESTINATION ${LATEX_PACKAGE_DIR})
install(DIRECTORY resources DESTINATION ${LATEX_PACKAGE_DIR})
