add_latex_document(
  test_beamer.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_latex_document(
  test_scrartcl.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_custom_target(tests_tumcaptions DEPENDS test_beamer test_scrartcl)
