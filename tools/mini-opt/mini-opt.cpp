//===- mini-opt.cpp - Mini Optimizer Driver -------------------===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
// Implements the `mini-opt` tool optimizer driver.
//
//===----------------------------------------------------------===//

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Mat/MatDialect.h"
#include "Mat/MatOps.h"
#include "Mat/Passes.h"

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<bool> enableTilePass("tile-pass",
                                           cl::desc("Enable the Tiling Pass"),
                                           cl::init(false));

/// Dumps MLIR.
int dumpMLIR();

// Entry point for the optimizer driver.
int main(int argc, char **argv) {
  // Register command line options and pass manager CL options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "Mini Toy Compiler\n");

  if (int error = dumpMLIR())
    return error;

  return 0;
}

int dumpMLIR() {
  mlir::MLIRContext context;
  // Load built-in dialects in this MLIR context.
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  // Load MatDialect.
  context.getOrLoadDialect<mlir::mat::MatDialect>();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
          llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input MLIR file.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file: " << inputFilename << "\n";
    return 3;
  }

  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  pm.addPass(mlir::createInlinerPass());

  // Create tiling pass if enabled.
  if (enableTilePass) {
    mlir::OpPassManager &optPm = pm.nest<mlir::func::FuncOp>();
    optPm.addPass(mlir::mat::createTilingPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;

  module->dump();
  return 0;
}