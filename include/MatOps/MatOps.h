// MatOps.h
// Created by Rajveer Singh on 07/08/24.
//

#ifndef MINITOYCOMPILER_MATOPS_H
#define MINITOYCOMPILER_MATOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "MatOps/MatOps.h.inc"

#endif //MINITOYCOMPILER_MATOPS_H