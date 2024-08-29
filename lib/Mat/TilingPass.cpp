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

  void runOnOperation() final;
};
} // namespace

void TilingPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Traverse within the function for each operation.
  func.walk([&](Operation *op) {
    if (auto addOp = dyn_cast<AddOp>(op)) {
      OpBuilder builder(addOp);

      SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(START_OFFSET),
                                           builder.getIndexAttr(START_OFFSET)};
      SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(TILE_SIZE),
                                         builder.getIndexAttr(TILE_SIZE)};
      SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                           builder.getIndexAttr(1)};

      FailureOr<TilingResult> tilingResult =
          addOp.getTiledImplementation(builder, offsets, sizes);
      if (failed(tilingResult)) {
        addOp.emitError("Tiling failed for AddOp");
        return signalPassFailure();
      }

      auto loc = addOp.getLoc();

      auto resultTensorType =
          addOp.getResult().getType().cast<RankedTensorType>();
      auto resultTensorShape = resultTensorType.getShape();

      Value resultTensor = builder.create<tensor::EmptyOp>(
          loc, resultTensorShape, resultTensorType.getElementType());

      // Current tile.
      Value *currentTile = tilingResult->tiledValues.begin();

      for (int64_t i = START_OFFSET; i < resultTensorShape[0]; i += TILE_SIZE) {
        for (int64_t j = START_OFFSET; j < resultTensorShape[1];
             j += TILE_SIZE, ++currentTile) {
          int64_t resultTileHeight =
              std::min(TILE_SIZE, resultTensorShape[0] - i);
          int64_t resultTileWidth =
              std::min(TILE_SIZE, resultTensorShape[1] - j);

          SmallVector<OpFoldResult> tiledResultOffsets = {
              builder.getIndexAttr(i), builder.getIndexAttr(j)};
          SmallVector<OpFoldResult> tiledResultSizes = {
              builder.getIndexAttr(resultTileHeight),
              builder.getIndexAttr(resultTileWidth)};

          // Insert the current tile to the result tensor.
          resultTensor = builder.create<tensor::InsertSliceOp>(
              loc, *currentTile, resultTensor, tiledResultOffsets,
              tiledResultSizes, strides);
        }
      }

      addOp.replaceAllUsesWith(resultTensor);
      addOp.erase();
    } else if (auto mulOp = dyn_cast<MulOp>(op)) {
      OpBuilder builder(mulOp);

      SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(START_OFFSET),
                                           builder.getIndexAttr(START_OFFSET)};
      SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(TILE_SIZE),
                                         builder.getIndexAttr(TILE_SIZE)};
      SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                           builder.getIndexAttr(1)};

      FailureOr<TilingResult> tilingResult =
          mulOp.getTiledImplementation(builder, offsets, sizes);
      if (failed(tilingResult)) {
        mulOp.emitError("Tiling failed for AddOp");
        return signalPassFailure();
      }

      auto loc = mulOp.getLoc();

      auto resultTensorType =
          mulOp.getResult().getType().cast<RankedTensorType>();
      auto resultTensorShape = resultTensorType.getShape();

      Value resultTensor = builder.create<tensor::EmptyOp>(
          loc, resultTensorShape, resultTensorType.getElementType());

      // Current tile.
      Value *currentTile = tilingResult->tiledValues.begin();

      for (int64_t i = START_OFFSET; i < resultTensorShape[0]; i += TILE_SIZE) {
        for (int64_t j = START_OFFSET; j < resultTensorShape[1];
             j += TILE_SIZE, ++currentTile) {
          int64_t resultTileHeight =
              std::min(TILE_SIZE, resultTensorShape[0] - i);
          int64_t resultTileWidth =
              std::min(TILE_SIZE, resultTensorShape[1] - j);

          SmallVector<OpFoldResult> tiledResultOffsets = {
              builder.getIndexAttr(i), builder.getIndexAttr(j)};
          SmallVector<OpFoldResult> tiledResultSizes = {
              builder.getIndexAttr(resultTileHeight),
              builder.getIndexAttr(resultTileWidth)};

          // Insert the current tile to the result tensor.
          resultTensor = builder.create<tensor::InsertSliceOp>(
              loc, *currentTile, resultTensor, tiledResultOffsets,
              tiledResultSizes, strides);
        }
      }

      mulOp.replaceAllUsesWith(resultTensor);
      mulOp.erase();
    }
  });
}

// Return the tiling pass.
std::unique_ptr<Pass> mat::createTilingPass() {
  return std::make_unique<TilingPass>();
}
