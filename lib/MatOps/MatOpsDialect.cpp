// MatOpsDialect.cpp
// Created by Rajveer Singh on 07/08/24.
//

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "MatOps/MatOpsDialect.h"
#include "MatOps/MatOps.h"

using namespace mlir;
using namespace mat;

#include "MatOps/MatOpsDialect.cpp.inc"

// Initialize MatOpsDialect and add operations.
void MatOpsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MatOps/MatOps.cpp.inc"
  >();
}

