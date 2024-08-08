# Add MLIR dialect library.
add_mlir_dialect_library(MLIRMatOps
        MatOpsDialect.cpp
        MatOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/MatOps

        DEPENDS
        MLIRMatOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
)