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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
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
    auto opName = op->getName().getStringRef();

    if (opName == "mat.add") {
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
    } else if (opName == "mat.mul") {
      lowerOpToLoops(
          op, operands, rewriter,
          [loc](OpBuilder &builder, ValueRange memRefOperands,
                ValueRange loopIvs) {
            // Generate an adaptor for the remapped operands of the
            // BinaryOp.
            typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

            auto lhsType =
                binaryAdaptor.getLhs().getType().template cast<ShapedType>();

            Value result = builder.create<arith::ConstantOp>(
                loc, builder.getIntegerAttr(lhsType.getElementType(), 0));

            // Create accumulator loop and pass result as loop-carried argument.
            auto accDim = lhsType.getShape()[1];
            auto accLoop = builder.create<affine::AffineForOp>(
                loc, /*lowerBound*/ 0, /*upperBound*/ accDim, /*step*/ 1,
                ValueRange{result});

            builder.setInsertionPointToStart(accLoop.getBody());

            // Generate loads for lhs and rhs at the inner loop.
            auto loadedLhs = builder.create<affine::AffineLoadOp>(
                loc, binaryAdaptor.getLhs(),
                ValueRange{loopIvs[0], accLoop.getInductionVar()});
            auto loadedRhs = builder.create<affine::AffineLoadOp>(
                loc, binaryAdaptor.getRhs(),
                ValueRange{accLoop.getInductionVar(), loopIvs[1]});

            Value product =
                builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
            Value newResult = builder.create<arith::AddIOp>(
                loc, accLoop.getRegionIterArgs()[0], product);

            // Yield accumulator result back to the loop.
            builder.create<affine::AffineYieldOp>(loc, newResult);

            builder.setInsertionPointAfter(accLoop);

            // Return the accumulated result.
            return accLoop.getResult(0);
          });
    }
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

//===----------------------------------------------------------===//
// MatToAffine RewritePatterns: Func operations
//===----------------------------------------------------------===//

struct FuncOpLowering : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    // Conversions are handled by the type converter
    // hence we can just replace it.
    auto funcOp = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(),
                                                op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), funcOp.getBody(), funcOp.end());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------===//
// MatToAffine RewritePatterns: Return operations
//===----------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // Conversions are handled by the type converter
    // hence we can just replace it.
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getOperands());
    return success();
  }
};

//===----------------------------------------------------------===//
// MatToAffine RewritePatterns: Call operations
//===----------------------------------------------------------===//

struct CallOpLowering : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const final {
    // Conversions are handled by the type converter
    // hence we can just replace it.
    SmallVector<Value, 4> operands = op.getOperands();
    auto callOp = rewriter.create<func::CallOp>(
        op.getLoc(), op.getCallee(), op.getResultTypes(), operands);
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------===//
// MatToAffine RewritePatterns: Extract Slice operations
//===----------------------------------------------------------===//

struct ExtractSliceOpLowering
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const final {
    // ...
    return success();
  }
};

//===----------------------------------------------------------===//
// MatToAffine RewritePatterns: Insert Slice operations
//===----------------------------------------------------------===//

struct InsertSliceOpLowering : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const final {
    // ...
    return success();
  }
};
} // namespace

//===----------------------------------------------------------===//
// MatToAffineTypeConverter
//===----------------------------------------------------------===//

/// A type converter to handle tensor to memref conversions
/// during partial lowering.
namespace {
struct MatToAffineTypeConverter : public TypeConverter {
  MatToAffineTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion(
        [](RankedTensorType type) { return convertTensorToMemRef(type); });
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
  MatToAffineTypeConverter typeConverter(&getContext());
  patterns.add<ConstantOpLowering, AddOpLowering, MulOpLowering, FuncOpLowering,
               ReturnOpLowering, CallOpLowering>(&getContext());

  // Populate all required conversion patterns with the type converter.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });

  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

  populateCallOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });

  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
           isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                            typeConverter) ||
           isLegalForReturnOpTypeConversionPattern(op, typeConverter);
  });

  // Apply partial conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

// Return the affine lowering pass.
std::unique_ptr<Pass> mat::createLowerToAffinePass() {
  return std::make_unique<MatToAffineLoweringPass>();
}
