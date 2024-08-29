# Add MLIR dialect
add_mlir_dialect(MatOps mat)

# Add MLIR docs.
add_mlir_doc(MatDialect MatDialect Mat/ -gen-dialect-doc)
add_mlir_doc(MatOps MatOps Mat/ -gen-op-doc)
