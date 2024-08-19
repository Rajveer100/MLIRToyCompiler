//===- MatOps.h - Matrix operation class ----------------------===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
//===----------------------------------------------------------===//
//
// Defines operation class for matrix.
//
//===----------------------------------------------------------===//

#ifndef MINITOYCOMPILER_MATOPS_H
#define MINITOYCOMPILER_MATOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#define GET_OP_CLASSES
#include "Mat/MatOps.h.inc"

#endif // MINITOYCOMPILER_MATOPS_H
