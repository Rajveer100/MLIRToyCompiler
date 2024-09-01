//===- LowerToLLVMPass.cpp - LLVM Lowering --------------------===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
//===----------------------------------------------------------===//
//
// Implements a pass to fully lower matrix operations
// to LLVM.
//
//===----------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "Mat/MatDialect.h"
#include "Mat/MatOps.h"
#include "Mat/Passes.h"

using namespace mlir;

//===----------------------------------------------------------===//
// MatToLLVMLoweringPass
//===----------------------------------------------------------===//

namespace {
struct MatToLLVMLoweringPass
    : public PassWrapper<MatToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatToLLVMLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect, arith::ArithDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void MatToLLVMLoweringPass::runOnOperation() {
  // Define conversion target.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  LLVMTypeConverter typeConverter(&getContext());

  // Define set of patterns to lower.
  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // Apply full conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

// Return the LLVM lowering pass.
std::unique_ptr<mlir::Pass> mat::createLowerToLLVMPass() {
  return std::make_unique<MatToLLVMLoweringPass>();
}
