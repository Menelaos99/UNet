# list to store loop generated targets
set(TUMFONTS_TARGETS "")

set(TEXTFONT_NAMES "helvet" "libertine" "palatino" "times")
set(MATHFONT_NAMES "ams" "pxtx")

latex_get_output_path(CMAKE_CURRENT_BINARY_DIR)

# default version with empty option strings
set(TEXTFONT "")
set(MATHFONT "")
configure_file("test_tumfonts.tex"
               "${CMAKE_CURRENT_BINARY_DIR}/test_tumfonts_default.tex"
)
add_latex_document(
  "test_tumfonts_default.tex"
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
  FORCE_PDF
)

# loop through options
foreach(TF IN LISTS TEXTFONT_NAMES)
  foreach(MF IN LISTS MATHFONT_NAMES)
    set(TEXTFONT "font=${TF}")
    set(MATHFONT "math=${MF}")
    set(CURRENT_LATEX_TARGET "test_tumfonts_${TF}_${MF}")
    list(APPEND TUMFONTS_TARGETS "${CURRENT_LATEX_TARGET}")
    configure_file("test_tumfonts.tex"
                   "${CMAKE_CURRENT_BINARY_DIR}/${CURRENT_LATEX_TARGET}.tex"
    )
    add_latex_document(
      "${CURRENT_LATEX_TARGET}.tex"
      INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
      FORCE_PDF
    )
  endforeach()
endforeach()

add_custom_target(
  tests_tumfonts DEPENDS test_tumfonts_default "${TUMFONTS_TARGETS}"
)
