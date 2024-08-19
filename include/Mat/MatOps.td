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

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

#endif // MAT_OPS
