# list to store loop generated targets
set(TUMCOLORS_TARGETS "")

set(COLORMODELS "rgb" "cmyk")

latex_get_output_path(CMAKE_CURRENT_BINARY_DIR)

foreach(COLORMODEL IN LISTS COLORMODELS)
  set(CURRENT_LATEX_TARGET "test_tumcolors_${COLORMODEL}")
  list(APPEND TUMCOLORS_TARGETS "${CURRENT_LATEX_TARGET}")
  configure_file("test_tumcolors.tex"
                 "${CMAKE_CURRENT_BINARY_DIR}/${CURRENT_LATEX_TARGET}.tex"
  )
  add_latex_document(
    "${CURRENT_LATEX_TARGET}.tex"
    INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
    FORCE_PDF
  )
endforeach()

add_custom_target(tests_tumcolors DEPENDS "${TUMCOLORS_TARGETS}")
