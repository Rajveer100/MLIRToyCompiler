//===- Passes.h - Matrix passes definitions -------------------===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
//===----------------------------------------------------------===//
//
// Entry point for all compiler passes.
//
//===----------------------------------------------------------===//

#ifndef MINITOYCOMPILER_PASSES_H
#define MINITOYCOMPILER_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::mat {
/// Create an instance of the tiling pass.
std::unique_ptr<mlir::Pass> createTilingPass();

/// Create an instance of lower to affine pass.
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

/// Create an instance of loewr to LLVM pass.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace mlir::mat

#endif // MINITOYCOMPILER_PASSES_H
