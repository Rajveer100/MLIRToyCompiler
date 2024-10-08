//===- MatDialect.cpp - Matrix dialect support code -----------===//
//
// Created by Rajveer Singh on 07/08/24.
//
//===----------------------------------------------------------===//

#include "Mat/MatDialect.h"
#include "Mat/MatOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

using namespace mlir;
using namespace mlir::mat;

#include "Mat/MatOpsDialect.cpp.inc"

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void MatDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mat/MatOps.cpp.inc"
      >();
}
