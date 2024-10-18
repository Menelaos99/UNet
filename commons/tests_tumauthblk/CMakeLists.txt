add_latex_document(
  test_authblk_many.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_latex_document(
  test_authblk_single.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_latex_document(
  test_authblk_three.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_latex_document(
  test_authblk_two.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_latex_document(
  test_authblk_two_two.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_custom_target(
  tests_tumauthblk
  DEPENDS test_authblk_many test_authblk_single test_authblk_three
          test_authblk_two test_authblk_two_two
)
