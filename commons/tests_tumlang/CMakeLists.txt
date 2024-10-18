add_latex_document(
  test_lang_default.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_latex_document(
  test_lang_english.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_latex_document(
  test_lang_german.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_latex_document(
  test_lang_language_english.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_latex_document(
  test_lang_language_german.tex
  INPUTS content.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

add_custom_target(
  tests_tumlang DEPENDS test_lang_default test_lang_german test_lang_english
                        test_lang_language_german test_lang_language_english
)
