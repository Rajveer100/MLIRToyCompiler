//===- TilingPass.cpp - Matrix Operations Tiling --------------===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
//===----------------------------------------------------------===//
//
// Implements an operation pass to implement tiling
// for the operations.
//
//===----------------------------------------------------------===//

#include <memory>

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Mat/MatDialect.h"
#include "Mat/MatOps.h"
#include "Mat/Passes.h"

using namespace mlir;
using namespace mat;

namespace {
/// Pass definition for tiling operations on matrices.
struct TilingPass
    : public mlir::PassWrapper<TilingPass, OperationPass<func::FuncOp>> {
  inline static constexpr int64_t START_OFFSET = 0;
  inline static constexpr int64_t TILE_SIZE = 8;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Traverse within the function for each operation.
    func.walk([&](AddOp op) {
      OpBuilder builder(op);

      // Set initial offsets, sizes and strides.
      SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(START_OFFSET),
                                           builder.getIndexAttr(START_OFFSET)};
      SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(TILE_SIZE),
                                         builder.getIndexAttr(TILE_SIZE)};
      SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                           builder.getIndexAttr(1)};

      // Compute tiling result.
      FailureOr<TilingResult> tilingResult =
          op.getTiledImplementation(builder, offsets, sizes);
      if (failed(tilingResult)) {
        op.emitError("Tiling failed.");
        return;
      }

      // Get the result type and create an empty result tensor.
      auto resultType = op.getType().cast<RankedTensorType>();
      Value resultTensor = builder.create<tensor::EmptyOp>(
          op.getLoc(), resultType.getShape(), resultType.getElementType());

      // Result tensor shape and size.
      auto resultTensorShape = resultType.getShape();
      SmallVector<OpFoldResult> resultTensorSize = {
          builder.getIndexAttr(resultTensorShape[0]),
          builder.getIndexAttr(resultTensorShape[1])};

      // Current tile.
      Value *currentTile = tilingResult->tiledValues.begin();

      // Insert result tile values in the result tensor.
      for (int64_t i = START_OFFSET; i < resultTensorShape[0]; i += TILE_SIZE) {
        for (int64_t j = START_OFFSET; j < resultTensorShape[1];
             j += TILE_SIZE) {
          SmallVector<OpFoldResult> currentOffset = {builder.getIndexAttr(i),
                                                     builder.getIndexAttr(j)};

          resultTensor = builder.create<tensor::InsertSliceOp>(
              op.getLoc(), *currentTile, resultTensor, currentOffset,
              resultTensorSize, strides);
          ++currentTile;
        }
      }

      // Replace all uses of this operation with the resulting tensor.
      op.replaceAllUsesWith(resultTensor);
      op.erase();
    });
  }
};
} // namespace

// Return the tiling pass.
std::unique_ptr<Pass> mat::createTilingPass() {
  return std::make_unique<TilingPass>();
}
