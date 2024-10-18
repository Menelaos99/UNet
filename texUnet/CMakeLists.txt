add_latex_document(
  unet.tex
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/commons
                      ${PROJECT_SOURCE_DIR}/letter
  FORCE_PDF
)


add_custom_target(
  texUnet DEPENDS unet
                          
)
