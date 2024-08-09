# Get MLIR dialect libs property.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# Set MLIR dialect libs.
set(LIBS
        ${dialect_libs}
        MLIRMat
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRLLVMDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIROptLib
)

# Add mini-opt executable.
add_llvm_executable(mini-opt mini-opt.cpp)

# Update compile flags and link libraries.
llvm_update_compile_flags(mini-opt)
target_link_libraries(mini-opt PRIVATE ${LIBS})

# Check link libraries for executable.
mlir_check_all_link_libraries(mini-opt)