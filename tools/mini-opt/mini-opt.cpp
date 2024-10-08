//===- mini-opt.cpp - Mini Optimizer Driver -------------------===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
// Implements the `mini-opt` tool optimizer driver.
//
//===----------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
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
#include "llvm/Support/raw_ostream.h"

#include "Mat/MatDialect.h"
#include "Mat/MatOps.h"
#include "Mat/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<bool> enableTilePass("tile-ops", cl::desc("Enable Tiling"),
                                    cl::init(false));

static cl::opt<bool> enableOneShotBufferize("one-shot-bufferize",
                                            cl::desc("Enable Bufferization"),
                                            cl::init(false));

static cl::opt<bool> enableLowerToAffine("lower-to-affine",
                                         cl::desc("Enable Affine Lowering"),
                                         cl::init(false));

static cl::opt<bool> enableLowerToLLVM("lower-to-llvm",
                                       cl::desc("Enable LLVM Lowering"),
                                       cl::init(false));

/// Loads MLIR.
int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module);

/// Dump LLVM IR.
int dumpLLVMIR(mlir::ModuleOp module);

// Entry point for the optimizer driver.
int main(int argc, char **argv) {
  // Register command line options and pass manager CL options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "Mini Toy Compiler\n");

  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);

  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();

  // Load MatDialect.
  context.getOrLoadDialect<mlir::mat::MatDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;

  if (int error = loadMLIR(context, module))
    return error;

  return 0;
}

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input MLIR file.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file: " << inputFilename << "\n";
    return 3;
  }

  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Create tiling pass if enabled.
  if (enableTilePass) {
    pm.addNestedPass<mlir::func::FuncOp>(mlir::mat::createTilingPass());
  }

  // Create affine lowering pass if enabled.
  if (enableLowerToAffine) {
    pm.addPass(mlir::mat::createLowerToAffinePass());
  }

  /// Create bufferization pass if enabled.
  if (enableOneShotBufferize) {
    pm.addPass(mlir::bufferization::createOneShotBufferizePass());
    pm.addPass(mlir::arith::createArithBufferizePass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tensor::createTensorBufferizePass());
  }

  // Apply optimizations.
  mlir::OpPassManager &optPm = pm.nest<mlir::func::FuncOp>();
  optPm.addPass(mlir::createCanonicalizerPass());
  optPm.addPass(mlir::createCSEPass());

  // Create LLVM lowering pass if enabled.
  if (enableLowerToLLVM) {
    pm.addPass(mlir::mat::createLowerToLLVMPass());
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Passes failed.\n";
    return 4;
  }

  if (enableLowerToLLVM) {
    if (int error = dumpLLVMIR(*module))
      return error;
  } else {
    module->dump();
  }

  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Configure the LLVM module.
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }

  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  module->print(llvm::outs());
  return 0;
}
