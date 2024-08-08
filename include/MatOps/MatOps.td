// MatOps.td - TableGen file defining operations for matrices.

include "MatOpsDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Element-wise matrix addition.
def MatOps_AddOp : MatOps_Op<"mat.add"> {
  let summary = "Element-wise addition of two int8_t matrices.";
  let description = [{
    This operation performs element-wise addition of two int8_t matrices
    and stores the result in a third matrix.
  }];

  let arguments = (ins
    TensorOf<[I8]>:$lhs,
    TensorOf<[I8]>:$rhs
  );

  let results = (outs
    TensorOf<[I8]>
  );
}

// Element-wise multiplication operation.
def MatOps_MulOp : MatOps_Op<"mat.mul"> {
  let summary = "Element-wise multiplication of two int8_t matrices";
  let description = [{
    This operation performs element-wise multiplication of two int8_t matrices
    and stores the result in a third matrix.
  }];

  let arguments = (ins
    TensorOf<[I8]>:$lhs,
    TensorOf<[I8]>:$rhs
  );

  let results = (outs
    TensorOf<[I8]>
  );
}