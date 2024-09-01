//===- LowerToAffinePass.cpp - Affine Lowering ----------------===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
//===----------------------------------------------------------===//
//
// Implements a pass to partially lower matrix operations
// to affine, memref and standard operations.
//
//===----------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

#include "Mat/MatDialect.h"
#include "Mat/MatOps.h"
#include "Mat/Passes.h"

using namespace mlir;

//===----------------------------------------------------------===//
// MatToAffine RewritePatterns
//===----------------------------------------------------------===//

/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Deallocate this alloc at the end of the block assuming no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration
/// of a lowered loop.
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = cast<RankedTensorType>(*op->result_type_begin());
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function returning value stored
        // at current index.
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                    ivs);
      });

  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------===//
// MatToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the
                     // BinaryOp.
                     typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                     // Generate loads for lhs and rhs at the inner loop.
                     auto loadedLhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getLhs(), loopIvs);
                     auto loadedRhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getRhs(), loopIvs);

                     // Create the binary operation performed.
                     return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                            loadedRhs);
                   });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<mat::AddOp, arith::AddIOp>;
using MulOpLowering = BinaryOpLowering<mat::MulOp, arith::MulIOp>;

//===----------------------------------------------------------===//
// MatToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<mat::ConstantOp> {
  using OpRewritePattern<mat::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mat::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    ElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // Allocate and assign the constant values.
    auto tensorType = cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // Generate constant indices up-to the largest dimension.
    // to avoid redundant computations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i :
           llvm::seq<int64_t>(0, *std::ranges::max_element(valueShape)))
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // Generate store operation recursively.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<IntegerAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // Base case.
      if (dimension == valueShape.size()) {
        rewriter.create<affine::AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            ArrayRef(indices));
        return;
      }

      // Iterate over the current dimension.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    storeElements(/*dimension=*/0);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------===//
// MatToAffineLoweringPass
//===----------------------------------------------------------===//

/// This pass partially lowers matrix operations that are computationaly
/// expensive to affine loops.
namespace {
struct MatToAffineLoweringPass
    : public PassWrapper<MatToAffineLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatToAffineLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void MatToAffineLoweringPass::runOnOperation() {
  // Define the conversion target.
  ConversionTarget target(getContext());

  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();

  target.addIllegalDialect<mat::MatDialect>();

  // Define set of patterns to lower.
  RewritePatternSet patterns(&getContext());
  patterns.add<ConstantOpLowering, AddOpLowering, MulOpLowering>(&getContext());

  // Apply partial conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

// Return the affine lowering pass.
std::unique_ptr<Pass> mat::createLowerToAffinePass() {
  return std::make_unique<MatToAffineLoweringPass>();
}
