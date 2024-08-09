# Add MLIR dialect library.
add_mlir_dialect_library(MLIRMat
        MatDialect.cpp
        MatOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Mat

        DEPENDS
        MLIRMatOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
)