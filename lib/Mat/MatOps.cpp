//===- MatOps.cpp - Matrix operation support code -----------===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
//===----------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "Mat/MatOps.h"
#include "Mat/MatDialect.h"

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
FailureOr<TilingResult> AddOp::getTiledImplementation(OpBuilder &builder,
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

  // Get the tensor types and dimensions
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

      auto currentTileShape = SmallVector<int64_t, 2>{actualTileHeight, actualTileWidth};
      auto subTensorType = RankedTensorType::get(currentTileShape, builder.getIntegerType(8));

      SmallVector<OpFoldResult> offsets = {
              builder.getIndexAttr(i),
              builder.getIndexAttr(j)
      };
      SmallVector<OpFoldResult> sizes = {
              builder.getIndexAttr(actualTileHeight),
              builder.getIndexAttr(actualTileWidth)
      };
      SmallVector<OpFoldResult> strides = {
              builder.getIndexAttr(1),
              builder.getIndexAttr(1)
      };

      // Extract the slices.
      auto subTensorLhs = builder.create<tensor::ExtractSliceOp>(loc, subTensorType, lhs,
                                                                 offsets, sizes, strides);
      auto subTensorRhs = builder.create<tensor::ExtractSliceOp>(loc, subTensorType, rhs,
                                                                 offsets, sizes, strides);

      // Perform the operation on the extracted slices.
      auto resultTile = builder.create<AddOp>(loc, subTensorType, subTensorLhs, subTensorRhs);

      // Insert the result back into the result tensor.
      Value resultTensor = builder.create<tensor::InsertSliceOp>(loc, resultTile,
                                                                 lhs, offsets, sizes, strides);

      // Store the tiled operation and result.
      tiledOps.push_back(resultTile);
      tiledValues.push_back(resultTensor);
    }
  }

  // Return the results of the tiling
  return TilingResult{tiledOps, tiledValues};
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
FailureOr<TilingResult> MulOp::getTiledImplementation(OpBuilder &builder,
                                                      ArrayRef<OpFoldResult> offsets,
                                                      ArrayRef<OpFoldResult> sizes) {
  // Todo: Needs little extra care.
}

void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }