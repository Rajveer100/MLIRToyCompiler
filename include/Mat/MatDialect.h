//===- MatDialect.h - Matrix dialect class --------------------===//
//
// Created by Rajveer Singh on 07/08/24.
//
//===----------------------------------------------------------===//
//
// Defines dialect class for matrix.
//
//===----------------------------------------------------------===//

#ifndef MINITOYCOMPILER_MATDIALECT_H
#define MINITOYCOMPILER_MATDIALECT_H

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Mat/MatOpsDialect.h.inc"

#endif //MINITOYCOMPILER_MATDIALECT_H