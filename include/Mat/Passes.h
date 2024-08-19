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
} // namespace mlir::mat

#endif // MINITOYCOMPILER_PASSES_H
