# Add MLIR dialect
add_mlir_dialect(MatOps mat)

# Add MLIR docs.
add_mlir_doc(MatOpsDialect MatOpsDialect MatOps/ -gen-dialect-doc)
add_mlir_doc(MatOps MatOps MatOps/ -gen-op-doc)