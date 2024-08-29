//===- MatOps.cpp - Matrix operation support code -----------===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
//===----------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "Mat/MatDialect.h"
#include "Mat/MatOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <sys/_types/_int64_t.h>
#include <sys/_types/_int8_t.h>

#define GET_OP_CLASSES
#include "Mat/MatOps.cpp.inc"

using namespace mlir;
using namespace mat;

/// A generalized parser for binary operations.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------===//

void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       int8_t value) {
  auto dataType = RankedTensorType::get({}, builder.getI8Type());
  auto dataAttribute = DenseIntElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseIntElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
  printer << getValue();
}

//===----------------------------------------------------------------===//
// ConstantOp: Bufferization
//===----------------------------------------------------------------===//

bool ConstantOp::bufferizesToMemoryRead(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return false;
}

bool ConstantOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return false;
}

bufferization::AliasingValueList
ConstantOp::getAliasingValues(OpOperand &opOperand,
                              const bufferization::AnalysisState &state) {
  bufferization::AliasingValueList aliasedValues;
  return aliasedValues;
}

mlir::LogicalResult ConstantOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options) {
  auto result = getResult();
  auto loc = getLoc();

  auto resultTensorType = result.getType().cast<RankedTensorType>();
  auto resultTensorShape = resultTensorType.getShape();

  // Allocate memory buffer of the appropriate size.
  auto memRefType =
      MemRefType::get(resultTensorShape, resultTensorType.getElementType());
  Value resultBuffer = rewriter.create<memref::AllocOp>(loc, memRefType);

  auto elementsAttr = getValue().cast<DenseIntElementsAttr>();
  for (auto index : llvm::enumerate(elementsAttr.getValues<IntegerAttr>())) {
    // Calculate row and column indices.
    int64_t flatIndex = index.index();
    int64_t rowIndex = flatIndex / resultTensorShape[1];
    int64_t colIndex = flatIndex % resultTensorShape[1];

    Value constantIndexRow =
        rewriter.create<arith::ConstantIndexOp>(loc, rowIndex);
    Value constantIndexCol =
        rewriter.create<arith::ConstantIndexOp>(loc, colIndex);

    Value scalarValue = rewriter.create<arith::ConstantOp>(loc, index.value());

    // Store in the result buffer.
    rewriter.create<memref::StoreOp>(
        loc, scalarValue, resultBuffer,
        ValueRange{constantIndexRow, constantIndexCol});
  }

  bufferization::replaceOpWithBufferizedValues(rewriter, *this, resultBuffer);
  return success();
}

//===----------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getI8Type()));
  state.addOperands({lhs, rhs});
}

LogicalResult AddOp::verify() {
  auto lhsShape = getLhs().getType().cast<RankedTensorType>().getShape();
  auto rhsShape = getRhs().getType().cast<RankedTensorType>().getShape();
  auto resultShape = getResult().getType().cast<RankedTensorType>().getShape();

  // Ensure operands have valid dimensions.
  if (lhsShape[0] != rhsShape[0] || lhsShape[1] != rhsShape[1])
    return emitOpError("invalid dimensions for matrix addition ")
           << lhsShape << " != " << rhsShape;

  if (resultShape[0] != lhsShape[0] || resultShape[1] != rhsShape[1])
    return emitOpError("invalid dimensions for result ")
           << "(" << resultShape[0] << " " << resultShape[1] << ")"
           << " != "
           << "(" << lhsShape[0] << " " << rhsShape[1] << ")";

  return success();
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------===//
// AddOp: Tiling
//===----------------------------------------------------------------===//

FailureOr<TilingResult>
AddOp::getTiledImplementation(OpBuilder &builder,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes) {
  // Ensure the sizes and offsets are valid
  if (offsets.size() != sizes.size() || offsets.size() != 2) {
    return failure();
  }

  SmallVector<int64_t> offsetValues, sizeValues;

  if (auto offsetValuesOpt = mlir::getConstantIntValues(offsets))
    offsetValues = *offsetValuesOpt;
  else
    return failure();

  if (auto sizeValuesOpt = mlir::getConstantIntValues(sizes))
    sizeValues = *sizeValuesOpt;
  else
    return failure();

  auto lhs = getLhs();
  auto rhs = getRhs();
  auto result = getResult();

  Location loc = getLoc();

  auto resultTensorType = result.getType().cast<RankedTensorType>();
  auto resultTensorShape = resultTensorType.getShape();

  // Tile dimensions.
  int64_t tileHeight = sizeValues[0];
  int64_t tileWidth = sizeValues[1];

  // Start offsets.
  int64_t startOffsetH = offsetValues[0];
  int64_t startOffsetW = offsetValues[1];

  SmallVector<Operation *> tiledOps;
  SmallVector<Value> tiledValues;

  // Generate tiles.
  for (int64_t i = startOffsetH; i < resultTensorShape[0]; i += tileHeight) {
    for (int64_t j = startOffsetW; j < resultTensorShape[1]; j += tileWidth) {
      // Take care of bounds.
      int64_t resultTileHeight = std::min(tileHeight, resultTensorShape[0] - i);
      int64_t resultTileWidth = std::min(tileWidth, resultTensorShape[1] - j);

      auto tiledResultTensorShape =
          SmallVector<int64_t, 2>{resultTileHeight, resultTileWidth};
      auto tiledResultTensorType = RankedTensorType::get(
          tiledResultTensorShape, builder.getIntegerType(8));

      SmallVector<OpFoldResult> tiledResultOffsets = {builder.getIndexAttr(i),
                                                      builder.getIndexAttr(j)};
      SmallVector<OpFoldResult> tiledResultSizes = {
          builder.getIndexAttr(resultTileHeight),
          builder.getIndexAttr(resultTileWidth)};
      SmallVector<OpFoldResult> tiledResultStrides = {builder.getIndexAttr(1),
                                                      builder.getIndexAttr(1)};

      // Extract the slices.
      auto tiledLhsTensor = builder.create<tensor::ExtractSliceOp>(
          loc, tiledResultTensorType, lhs, tiledResultOffsets, tiledResultSizes,
          tiledResultStrides);
      auto tiledRhsTensor = builder.create<tensor::ExtractSliceOp>(
          loc, tiledResultTensorType, rhs, tiledResultOffsets, tiledResultSizes,
          tiledResultStrides);

      // Perform the operation on the extracted slices.
      auto tiledResult = builder.create<AddOp>(loc, tiledResultTensorType,
                                               tiledLhsTensor, tiledRhsTensor);

      // Store the tiled operation and result.
      tiledOps.push_back(tiledResult.getOperation());
      tiledValues.push_back(tiledResult);
    }
  }
  return TilingResult{tiledOps, tiledValues};
}

//===----------------------------------------------------------------===//
// AddOp: Bufferization
//===----------------------------------------------------------------===//

bool AddOp::bufferizesToMemoryRead(OpOperand &opOperand,
                                   const bufferization::AnalysisState &state) {
  return false;
}

bool AddOp::bufferizesToMemoryWrite(OpOperand &opOperand,
                                    const bufferization::AnalysisState &state) {
  return false;
}

bufferization::AliasingValueList
AddOp::getAliasingValues(OpOperand &opOperand,
                         const bufferization::AnalysisState &state) {
  bufferization::AliasingValueList aliasedValues;
  return aliasedValues;
}

mlir::LogicalResult
AddOp::bufferize(mlir::RewriterBase &rewriter,
                 const mlir::bufferization::BufferizationOptions &options) {
  auto result = getResult();
  auto loc = getLoc();

  auto resultTensorType = result.getType().cast<RankedTensorType>();
  auto resultTensorShape = resultTensorType.getShape();

  // Get operand buffers.
  auto lhsBuffer = bufferization::getBuffer(rewriter, getLhs(), options);
  auto rhsBuffer = bufferization::getBuffer(rewriter, getRhs(), options);

  // Allocate memory buffer of the appropriate size.
  auto memRefType =
      MemRefType::get(resultTensorShape, resultTensorType.getElementType());
  Value resultBuffer = rewriter.create<memref::AllocOp>(loc, memRefType);

  for (int64_t i = 0; i < resultTensorShape[0]; ++i) {
    for (int64_t j = 0; j < resultTensorShape[1]; ++j) {
      Value constantIndexRow = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value constantIndexCol = rewriter.create<arith::ConstantIndexOp>(loc, j);

      // Load the lhs and rhs elements.
      auto lhsElement = rewriter.create<memref::LoadOp>(
          loc, *lhsBuffer, ValueRange{constantIndexRow, constantIndexCol});
      auto rhsElement = rewriter.create<memref::LoadOp>(
          loc, *rhsBuffer, ValueRange{constantIndexRow, constantIndexCol});

      // Perform the operation.
      auto resultElement =
          rewriter.create<arith::AddIOp>(loc, lhsElement, rhsElement);

      // Store the result.
      rewriter.create<memref::StoreOp>(
          loc, resultElement, resultBuffer,
          ValueRange{constantIndexRow, constantIndexCol});
    }
  }

  bufferization::replaceOpWithBufferizedValues(rewriter, *this, resultBuffer);
  return success();
}

//===----------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getI8Type()));
  state.addOperands({lhs, rhs});
}

LogicalResult MulOp::verify() {
  auto lhsShape = getLhs().getType().cast<RankedTensorType>().getShape();
  auto rhsShape = getRhs().getType().cast<RankedTensorType>().getShape();
  auto resultShape = getResult().getType().cast<RankedTensorType>().getShape();

  // Ensure operands have valid dimensions.
  if (lhsShape[1] != rhsShape[0])
    return emitOpError("invalid dimensions for lhs and rhs ")
           << lhsShape[1] << " != " << rhsShape[0];

  if (resultShape[0] != lhsShape[0] || resultShape[1] != rhsShape[1])
    return emitOpError("invalid dimensions for result ")
           << "(" << resultShape[0] << " " << resultShape[1] << ")"
           << " != "
           << "(" << lhsShape[0] << " " << rhsShape[1] << ")";

  return success();
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------===//
// MulOp: Tiling
//===----------------------------------------------------------------===//

FailureOr<TilingResult>
MulOp::getTiledImplementation(OpBuilder &builder,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes) {
  // Ensure the sizes and offsets are valid
  if (offsets.size() != sizes.size() || offsets.size() != 2) {
    return failure();
  }

  SmallVector<int64_t> offsetValues, sizeValues;

  if (auto offsetValuesOpt = mlir::getConstantIntValues(offsets))
    offsetValues = *offsetValuesOpt;
  else
    return failure();

  if (auto sizeValuesOpt = mlir::getConstantIntValues(sizes))
    sizeValues = *sizeValuesOpt;
  else
    return failure();

  auto lhs = getLhs();
  auto rhs = getRhs();
  auto result = getResult();

  Location loc = getLoc();

  // Get the tensor types and dimensions.
  auto lhsTensorType = lhs.getType().cast<RankedTensorType>();
  auto rhsTensorType = rhs.getType().cast<RankedTensorType>();
  auto resultTensorType = result.getType().cast<RankedTensorType>();

  auto lhsTensorShape = lhsTensorType.getShape();
  auto rhsTensorShape = rhsTensorType.getShape();
  auto resultTensorShape = resultTensorType.getShape();

  // Tile dimensions.
  int64_t tileHeight = sizeValues[0];
  int64_t tileWidth = sizeValues[1];

  // Start offsets.
  int64_t startOffsetH = offsetValues[0];
  int64_t startOffsetW = offsetValues[1];

  SmallVector<Operation *> tiledOps;
  SmallVector<Value> tiledValues;

  // Generate tiles.
  for (int64_t i = startOffsetH; i < resultTensorShape[0]; i += tileHeight) {
    for (int64_t j = startOffsetW; j < resultTensorShape[1]; j += tileWidth) {
      // Take care of bounds.
      int64_t lhsTileHeight = std::min(tileHeight, lhsTensorShape[0] - i);
      int64_t lhsTileWidth = lhsTensorShape[1];

      int64_t rhsTileHeight = lhsTensorShape[1];
      int64_t rhsTileWidth = std::min(tileWidth, rhsTensorShape[1] - j);

      int64_t resultTileHeight = std::min(tileHeight, lhsTensorShape[0] - i);
      int64_t resultTileWidth = std::min(tileWidth, rhsTensorShape[1] - j);

      auto tiledLhsShape = SmallVector<int64_t, 2>{lhsTileHeight, lhsTileWidth};
      auto tiledLhsTensorType =
          RankedTensorType::get(tiledLhsShape, builder.getIntegerType(8));

      auto tiledRhsShape = SmallVector<int64_t, 2>{rhsTileHeight, rhsTileWidth};
      auto tiledRhsTensorType =
          RankedTensorType::get(tiledRhsShape, builder.getIntegerType(8));

      auto tiledResultShape =
          SmallVector<int64_t, 2>{resultTileHeight, resultTileWidth};
      auto tiledResultTensorType =
          RankedTensorType::get(tiledResultShape, builder.getIntegerType(8));

      SmallVector<OpFoldResult> tiledLhsOffsets = {builder.getIndexAttr(i),
                                                   builder.getIndexAttr(0)};
      SmallVector<OpFoldResult> tiledLhsSizes = {
          builder.getIndexAttr(lhsTileHeight),
          builder.getIndexAttr(lhsTileWidth)};

      SmallVector<OpFoldResult> tiledRhsOffsets = {builder.getIndexAttr(0),
                                                   builder.getIndexAttr(j)};
      SmallVector<OpFoldResult> tiledRhsSizes = {
          builder.getIndexAttr(rhsTileHeight),
          builder.getIndexAttr(rhsTileWidth)};

      SmallVector<OpFoldResult> tiledResultOffsets = {builder.getIndexAttr(i),
                                                      builder.getIndexAttr(j)};
      SmallVector<OpFoldResult> tiledResultSizes = {
          builder.getIndexAttr(resultTileHeight),
          builder.getIndexAttr(resultTileWidth)};

      SmallVector<OpFoldResult> tiledStrides = {builder.getIndexAttr(1),
                                                builder.getIndexAttr(1)};

      // Extract the slices.
      auto tiledTensorLhs = builder.create<tensor::ExtractSliceOp>(
          loc, tiledLhsTensorType, lhs, tiledLhsOffsets, tiledLhsSizes,
          tiledStrides);
      auto tiledTensorRhs = builder.create<tensor::ExtractSliceOp>(
          loc, tiledRhsTensorType, rhs, tiledRhsOffsets, tiledRhsSizes,
          tiledStrides);

      // Perform the operation on the extracted slices.
      auto tiledResult = builder.create<MulOp>(loc, tiledResultTensorType,
                                               tiledTensorLhs, tiledTensorRhs);

      // Store the tiled operation and result.
      tiledOps.push_back(tiledResult.getOperation());
      tiledValues.push_back(tiledResult);
    }
  }
  return TilingResult{tiledOps, tiledValues};
}

//===----------------------------------------------------------------===//
// MulOp: Bufferization
//===----------------------------------------------------------------===//

bool MulOp::bufferizesToMemoryRead(OpOperand &opOperand,
                                   const bufferization::AnalysisState &state) {
  return false;
}

bool MulOp::bufferizesToMemoryWrite(OpOperand &opOperand,
                                    const bufferization::AnalysisState &state) {
  return false;
}

bufferization::AliasingValueList
MulOp::getAliasingValues(OpOperand &opOperand,
                         const bufferization::AnalysisState &state) {
  bufferization::AliasingValueList aliasedValues;
  return aliasedValues;
}

mlir::LogicalResult
MulOp::bufferize(mlir::RewriterBase &rewriter,
                 const mlir::bufferization::BufferizationOptions &options) {
  auto result = getResult();
  auto loc = getLoc();

  auto resultTensorType = result.getType().cast<RankedTensorType>();
  auto resultTensorShape = resultTensorType.getShape();

  // Get operand buffers.
  auto lhsBuffer = bufferization::getBuffer(rewriter, getLhs(), options);
  auto rhsBuffer = bufferization::getBuffer(rewriter, getRhs(), options);

  // Allocate memory buffer of the appropriate size.
  auto memRefType =
      MemRefType::get(resultTensorShape, resultTensorType.getElementType());
  Value resultBuffer = rewriter.create<memref::AllocOp>(loc, memRefType);

  for (int64_t i = 0; i < resultTensorShape[0]; ++i) {
    for (int64_t j = 0; j < resultTensorShape[1]; ++j) {
      Value constantIndexRow = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value constantIndexCol = rewriter.create<arith::ConstantIndexOp>(loc, j);

      auto lhsElement = rewriter.create<memref::LoadOp>(
          loc, *lhsBuffer, ValueRange{constantIndexRow, constantIndexCol});
      auto rhsElement = rewriter.create<memref::LoadOp>(
          loc, *rhsBuffer, ValueRange{constantIndexRow, constantIndexCol});

      // Perform the operation.
      auto resultElement =
          rewriter.create<arith::MulIOp>(loc, lhsElement, rhsElement);

      // Store the result.
      rewriter.create<memref::StoreOp>(
          loc, resultElement, resultBuffer,
          ValueRange{constantIndexRow, constantIndexCol});
    }
  }

  bufferization::replaceOpWithBufferizedValues(rewriter, *this, resultBuffer);
  return success();
}
