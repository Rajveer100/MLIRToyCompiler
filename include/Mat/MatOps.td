//===- MatOps.td - Matrix operations definitions -----*- tablegen -*-===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
//===----------------------------------------------------------------===//
//
// Defines element-wise matrix operations.
//
//===----------------------------------------------------------------===//

#ifndef MAT_OPS
#define MAT_OPS

include "MatDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"

// Base class for matrix operations.
class Mat_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mat_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------===//
def ConstantOp : Mat_Op<"constant", 
    [Pure, 
    DeclareOpInterfaceMethods<BufferizableOpInterface, ["bufferizesToMemoryRead", 
                                                        "bufferizesToMemoryWrite", 
                                                        "getAliasingValues", 
                                                        "bufferize"]>]> {
  let summary = "constant";
  let description = [{
    This operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. 
  }];

  let arguments = (ins ElementsAttr:$value);

  let results = (outs TensorOf<[I8]>);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "DenseIntElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "int8_t":$value)>
  ];
}

//===----------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------===//
def Mat_AddOp : Mat_Op<"add",
    [DeclareOpInterfaceMethods<TilingInterface, ["getTiledImplementation"]>, 
    DeclareOpInterfaceMethods<BufferizableOpInterface, ["bufferizesToMemoryRead", 
                                                        "bufferizesToMemoryWrite", 
                                                        "getAliasingValues", 
                                                        "bufferize"]>]> {
  let summary = "Element-wise addition of two int8_t matrices.";
  let description = [{
    This operation performs element-wise addition of two int8_t matrices
    and stores the result in a third matrix.

    Both matrices must have the same dimensions, i.e, for a valid operation:

    dim(A) = dim(B) = (n x m)

    Example:

      ```
      %0 = mat.constant dense<1> : tensor<2x4xi8>
      %1 = mat.constant dense<1> : tensor<2x4xi8>

      %2 = mat.add %0, %1
            : (tensor<2x4xi8>, tensor<2x4xi8>) -> tensor<2x4xi8>
      ```
  }];

  let arguments = (ins
    TensorOf<[I8]>:$lhs,
    TensorOf<[I8]>:$rhs
  );

  let results = (outs
    TensorOf<[I8]>
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------===//
def Mat_MulOp : Mat_Op<"mul",
    [DeclareOpInterfaceMethods<TilingInterface, ["getTiledImplementation"]>,
    DeclareOpInterfaceMethods<BufferizableOpInterface, ["bufferizesToMemoryRead", 
                                                        "bufferizesToMemoryWrite", 
                                                        "getAliasingValues", 
                                                        "bufferize"]>]> {
  let summary = "Element-wise multiplication of two int8_t matrices";
  let description = [{
    This operation performs multiplication (dot-product) of two int8_t matrices
    and stores the result in a third matrix.

    The number of columns of first matrix must be equal to the number of rows
    of the second matrix, i.e, for a valid operation:

    dim(A) = (n1 x m1)
    dim(B) = (n2 x m2)

    => m1 = n2

    Example:

      ```
      %0 = mat.constant dense<1> : tensor<2x4xi8>
      %1 = mat.constant dense<1> : tensor<4x2xi8>

      %2 = mat.mul %0, %1
            : (tensor<2x4xi8>, tensor<4x2xi8>) -> tensor<2x2xi8>
      ```
  }];

  let arguments = (ins
    TensorOf<[I8]>:$lhs,
    TensorOf<[I8]>:$rhs
  );

  let results = (outs
    TensorOf<[I8]>
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];

  let hasVerifier = 1;
}

#endif // MAT_OPS
