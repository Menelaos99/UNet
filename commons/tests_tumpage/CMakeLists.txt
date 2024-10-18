# list to store loop generated targets
set(TUMPAGE_TARGETS "")

set(PAPER_NAMES "a0paper" "a1paper" "a2paper")
list(APPEND PAPER_NAMES "a3paper" "a4paper" "a5paper")
set(PAPER_ORIENTATIONS "portrait" "landscape")

latex_get_output_path(CMAKE_CURRENT_BINARY_DIR)

foreach(PAPER IN LISTS PAPER_NAMES)
  foreach(ORIENTATION IN LISTS PAPER_ORIENTATIONS)
    set(CURRENT_LATEX_TARGET "test_tumpage_${PAPER}_${ORIENTATION}")
    list(APPEND TUMPAGE_TARGETS "${CURRENT_LATEX_TARGET}")
    configure_file("test_tumpage.tex"
                   "${CMAKE_CURRENT_BINARY_DIR}/${CURRENT_LATEX_TARGET}.tex"
    )
    add_latex_document(
      "${CURRENT_LATEX_TARGET}.tex"
      INPUTS reference_values.tex
      INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
      FORCE_PDF
    )
  endforeach()
endforeach()

add_latex_document(
  test_tumpage_letter.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_latex_document(
  test_tumpage_brochure_a4.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_custom_target(
  tests_tumpage DEPENDS test_tumpage_letter test_tumpage_brochure_a4
                        "${TUMPAGE_TARGETS}"
)
