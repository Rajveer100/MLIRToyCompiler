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
#include "mlir/Support/LogicalResult.h"
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

/// Return true if the buffer of the given tensor OpOperand is read.
bool ConstantOp::bufferizesToMemoryRead(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return false;
}

/// Return true if the buffer of the given tensor OpOperand is written (if
/// bufferizing in-place).
bool ConstantOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return false;
}

/// Return the Values that may alias with a given OpOperand when bufferized
/// in-place.
bufferization::AliasingValueList
ConstantOp::getAliasingValues(OpOperand &opOperand,
                              const bufferization::AnalysisState &state) {
  bufferization::AliasingValueList aliasedValues;
  return aliasedValues;
}

/// Bufferize `ConstantOp` operation by rewriting the op with the given
/// rewriter.
mlir::LogicalResult ConstantOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options) {
  // Get the operation and result type.
  auto constantOp = cast<ConstantOp>(*this);
  auto tensorType = constantOp.getResult().getType().cast<RankedTensorType>();
  auto tensorShape = tensorType.getShape();

  // Allocate memory buffer of the appropriate size.
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  Value buffer =
      rewriter.create<memref::AllocOp>(constantOp.getLoc(), memrefType);

  // Get the elements.
  auto elementsAttr = constantOp.getValue().cast<DenseIntElementsAttr>();

  // Store the constant data into the allocated buffer
  for (auto index : llvm::enumerate(elementsAttr.getValues<IntegerAttr>())) {
    // Calculate row and column indices.
    int64_t flatIndex = index.index();
    int64_t rowIndex = flatIndex / tensorShape[1];
    int64_t colIndex = flatIndex % tensorShape[1];

    Value constantIndexRow =
        rewriter.create<arith::ConstantIndexOp>(constantOp.getLoc(), rowIndex);
    Value constantIndexCol =
        rewriter.create<arith::ConstantIndexOp>(constantOp.getLoc(), colIndex);

    Value scalarValue =
        rewriter.create<arith::ConstantOp>(constantOp.getLoc(), index.value());
    rewriter.create<memref::StoreOp>(
        constantOp.getLoc(), scalarValue, buffer,
        ValueRange{constantIndexRow, constantIndexCol});
  }

  // Replace the result of the operation with the buffer.
  bufferization::replaceOpWithBufferizedValues(rewriter, *this, buffer);

  return success();
}

void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
  printer << getValue();
}

//===----------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------===//
void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getI8Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

/// Returns tiled implementation of the `AddOp` operation.
FailureOr<TilingResult>
AddOp::getTiledImplementation(OpBuilder &builder,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes) {
  // Ensure the sizes and offsets are valid
  if (offsets.size() != sizes.size() || offsets.size() != 2) {
    return failure();
  }

  SmallVector<int64_t> offsetValues, sizeValues;

  // Extract offsets from fold results.
  if (auto offsetValuesOpt = mlir::getConstantIntValues(offsets))
    offsetValues = *offsetValuesOpt;
  else
    return failure();

  // Extract sizes from fold results.
  if (auto sizeValuesOpt = mlir::getConstantIntValues(sizes))
    sizeValues = *sizeValuesOpt;
  else
    return failure();

  // Create vectors to hold the tiled values and operations
  SmallVector<Operation *> tiledOps;
  SmallVector<Value> tiledValues;

  // Get tensor operands and loc.
  Value lhs = getOperand(0);
  Value rhs = getOperand(1);
  Location loc = getLoc();

  // Get the tensor types and dimensions.
  auto tensorType = lhs.getType().cast<RankedTensorType>();
  auto tensorShape = tensorType.getShape();

  // Tile dimensions.
  int64_t tileHeight = sizeValues[0];
  int64_t tileWidth = sizeValues[1];

  // Start offsets.
  int64_t startOffsetH = offsetValues[0];
  int64_t startOffsetW = offsetValues[1];

  // Create a loop to generate tiles
  for (int64_t i = startOffsetH; i < tensorShape[0]; i += tileHeight) {
    for (int64_t j = startOffsetW; j < tensorShape[1]; j += tileWidth) {
      // Take care of bounds.
      int64_t actualTileHeight = std::min(tileHeight, tensorShape[0] - i);
      int64_t actualTileWidth = std::min(tileWidth, tensorShape[1] - j);

      auto currentTileShape =
          SmallVector<int64_t, 2>{actualTileHeight, actualTileWidth};
      auto subTensorType =
          RankedTensorType::get(currentTileShape, builder.getIntegerType(8));

      SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(i),
                                           builder.getIndexAttr(j)};
      SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(actualTileHeight),
                                         builder.getIndexAttr(actualTileWidth)};
      SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                           builder.getIndexAttr(1)};

      // Extract the slices.
      auto subTensorLhs = builder.create<tensor::ExtractSliceOp>(
          loc, subTensorType, lhs, offsets, sizes, strides);
      auto subTensorRhs = builder.create<tensor::ExtractSliceOp>(
          loc, subTensorType, rhs, offsets, sizes, strides);

      // Perform the operation on the extracted slices.
      auto resultTile =
          builder.create<AddOp>(loc, subTensorType, subTensorLhs, subTensorRhs);

      // Insert the result back into the result tensor.
      Value resultTensor = builder.create<tensor::InsertSliceOp>(
          loc, resultTile, lhs, offsets, sizes, strides);

      // Store the tiled operation and result.
      tiledOps.push_back(resultTile);
      tiledValues.push_back(resultTensor);
    }
  }

  // Return the results of the tiling
  return TilingResult{tiledOps, tiledValues};
}

/// Return true if the buffer of the given tensor OpOperand is read.
bool AddOp::bufferizesToMemoryRead(OpOperand &opOperand,
                                   const bufferization::AnalysisState &state) {
  return false;
}

/// Return true if the buffer of the given tensor OpOperand is written (if
/// bufferizing in-place).
bool AddOp::bufferizesToMemoryWrite(OpOperand &opOperand,
                                    const bufferization::AnalysisState &state) {
  return false;
}

/// Return the Values that may alias with a given OpOperand when bufferized
/// in-place.
bufferization::AliasingValueList
AddOp::getAliasingValues(OpOperand &opOperand,
                         const bufferization::AnalysisState &state) {
  bufferization::AliasingValueList aliasedValues;
  return aliasedValues;
}

/// Bufferize `AddOp` operation by rewriting the op with the given rewriter.
mlir::LogicalResult
AddOp::bufferize(mlir::RewriterBase &rewriter,
                 const mlir::bufferization::BufferizationOptions &options) {
  // Get the operation and result type.
  auto addOp = cast<AddOp>(*this);

  // Get operands and result.
  auto lhs = addOp.getOperand(0);
  auto rhs = addOp.getOperand(1);
  auto result = addOp.getResult();

  // Get tensor type and shape.
  auto tensorType = result.getType().cast<RankedTensorType>();
  auto tensorShape = tensorType.getShape();

  // Get operand buffers.
  auto lhsBuffer = bufferization::getBuffer(rewriter, lhs, options);
  auto rhsBuffer = bufferization::getBuffer(rewriter, rhs, options);

  // Allocate memory buffer of the appropriate size.
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  Value resultBuffer =
      rewriter.create<memref::AllocOp>(addOp.getLoc(), memrefType);

  for (int64_t i = 0; i < tensorShape[0]; ++i) {
    for (int64_t j = 0; j < tensorShape[1]; ++j) {
      Value constantIndexRow =
          rewriter.create<arith::ConstantIndexOp>(addOp.getLoc(), i);
      Value constantIndexCol =
          rewriter.create<arith::ConstantIndexOp>(addOp.getLoc(), j);

      auto lhsElement = rewriter.create<memref::LoadOp>(
          addOp.getLoc(), *lhsBuffer,
          ValueRange{constantIndexRow, constantIndexCol});
      auto rhsElement = rewriter.create<memref::LoadOp>(
          addOp.getLoc(), *rhsBuffer,
          ValueRange{constantIndexRow, constantIndexCol});

      auto resultElement = rewriter.create<arith::AddIOp>(
          addOp.getLoc(), lhsElement, rhsElement);

      rewriter.create<memref::StoreOp>(
          addOp.getLoc(), resultElement, resultBuffer,
          ValueRange{constantIndexRow, constantIndexCol});
    }
  }

  // Replace the result of the operation with the buffer.
  bufferization::replaceOpWithBufferizedValues(rewriter, *this, resultBuffer);
  return success();
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------===//
void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getI8Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

/// Returns tiled implementation of the `MulOp` operation.
FailureOr<TilingResult>
MulOp::getTiledImplementation(OpBuilder &builder,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes) {
  // Todo: Needs little extra care.
  return success();
}

/// Return true if the buffer of the given tensor OpOperand is read.
bool MulOp::bufferizesToMemoryRead(OpOperand &opOperand,
                                   const bufferization::AnalysisState &state) {
  return false;
}

/// Return true if the buffer of the given tensor OpOperand is written (if
/// bufferizing in-place).
bool MulOp::bufferizesToMemoryWrite(OpOperand &opOperand,
                                    const bufferization::AnalysisState &state) {
  return false;
}

/// Return the Values that may alias with a given OpOperand when bufferized
/// in-place.
bufferization::AliasingValueList
MulOp::getAliasingValues(OpOperand &opOperand,
                         const bufferization::AnalysisState &state) {
  bufferization::AliasingValueList aliasedValues;
  return aliasedValues;
}

/// Bufferize `MulOp` operation by rewriting the op with the given rewriter.
mlir::LogicalResult
MulOp::bufferize(mlir::RewriterBase &rewriter,
                 const mlir::bufferization::BufferizationOptions &options) {
  // Todo: Needs little extra care.
  return success();
}

void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }
