//===----------------------------------------------------------------------===//
//
// Author: Rajveer <rajveer.developer@icloud.com>
//
// Test matrix arithmetic operations.
//===----------------------------------------------------------------------===//

// Ideal case when dimensions are perfectly divisible.
func.func @mat_arithmetic() -> tensor<24x16xi8> {
  %0 = arith.constant dense<1> : tensor<24x32xi8>
  %1 = arith.constant dense<1> : tensor<32x16xi8>
  %2 = arith.constant dense<1> : tensor<24x16xi8>
  %3 = mat.mul %0, %1
        : (tensor<24x32xi8>, tensor<32x16xi8>) -> tensor<24x16xi8>
  %4 = mat.add %3, %2
        : (tensor<24x16xi8>, tensor<24x16xi8>) -> tensor<24x16xi8>
  func.return %4 : tensor<24x16xi8>
}

// Check for non-ideal cases.
func.func @mat_arithmetic1() -> tensor<23x11xi8> {
  %0 = arith.constant dense<1> : tensor<23x32xi8>
  %1 = arith.constant dense<1> : tensor<32x11xi8>
  %2 = arith.constant dense<1> : tensor<23x11xi8>
  %3 = mat.mul %0, %1
        : (tensor<23x32xi8>, tensor<32x11xi8>) -> tensor<23x11xi8>
  %4 = mat.add %3, %2
        : (tensor<23x11xi8>, tensor<23x11xi8>) -> tensor<23x11xi8>
  func.return %4 : tensor<23x11xi8>
}