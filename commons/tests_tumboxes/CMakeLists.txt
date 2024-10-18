add_latex_document(
  test_tumboxes_beamer.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_latex_document(
  test_tumboxes_beamer_default.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_custom_target(
  tests_tumboxes DEPENDS test_tumboxes_beamer test_tumboxes_beamer_default
)
