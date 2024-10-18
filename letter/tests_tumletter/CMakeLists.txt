add_latex_document(
  test_tumletter_default.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
                      ${PROJECT_SOURCE_DIR}/letter
  FORCE_PDF
)

add_latex_document(
  test_tumletter_longfoot.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
                      ${PROJECT_SOURCE_DIR}/letter
  FORCE_PDF
)

add_latex_document(
  test_tumletter_invoice.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
                      ${PROJECT_SOURCE_DIR}/letter
  FORCE_PDF
)

add_custom_target(
  tests_tumletter DEPENDS test_tumletter_default test_tumletter_longfoot
                          test_tumletter_invoice
)
