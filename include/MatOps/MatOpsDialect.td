// MatOpsDialect.td - TableGen file defining dialect for matrix operations.

include "mlir/IR/OpBase.td"

// MatOps Dialect definition.
def MatOps_Dialect : Dialect {
  let name = "mat";

  let summary = "A high-level dialect for defining matrix operations.";
  let description = [{
    This dialect provides a representation of matrix operations
    allowing to compute element-wise addition and multiplication
    with analysis and optimization.
  }];

  let cppNamespace = "::mat";
}

// Base class for MatOps dialect operations.
class MatOps_Op<string mnemonic, list<Trait> traits = []> :
        Op<MatOps_Dialect, mnemonic, traits>;