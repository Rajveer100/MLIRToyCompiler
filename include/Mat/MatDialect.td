//===- MatDialect.td - Matrix dialect definition -----*- tablegen -*-===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
//===----------------------------------------------------------------===//
//
// Defines dialect for matrix.
//
//===----------------------------------------------------------------===//

#ifndef MAT_DIALECT
#define MAT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------===//
// Mat Dialect
//===----------------------------------------------------------------===//
def Mat_Dialect : Dialect {
  let name = "mat";

  let summary = "A high-level dialect for defining matrix operations.";
  let description = [{
    This dialect provides a representation of matrix operations
    allowing to compute addition and multiplication
    with analysis and optimization.
  }];

  let cppNamespace = "::mlir::mat";
}

#endif // MAT_DIALECT
