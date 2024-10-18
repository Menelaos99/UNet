# list to store loop generated targets
set(TUMMATH_TARGETS "")

set(TEXTFONT_NAMES "helvet" "libertine" "palatino" "times")

latex_get_output_path(CMAKE_CURRENT_BINARY_DIR)

foreach(TEXTFONT IN LISTS TEXTFONT_NAMES)
  set(CURRENT_LATEX_TARGET "test_tummath_${TEXTFONT}")
  list(APPEND TUMMATH_TARGETS "${CURRENT_LATEX_TARGET}")
  configure_file("test_tummath.tex"
                 "${CMAKE_CURRENT_BINARY_DIR}/${CURRENT_LATEX_TARGET}.tex"
  )
  add_latex_document(
    "${CURRENT_LATEX_TARGET}.tex"
    INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
    FORCE_PDF
  )
endforeach()

add_custom_target(tests_tummath DEPENDS "${TUMMATH_TARGETS}")
