// See cuda_runtime.h in this directory: compile-check stand-in, not real cuBLAS.
// The signatures match the real ones so a wrong argument order or count in the
// GEMM calls is caught here rather than at the first GPU build.
#pragma once

#include <cuda_runtime.h>

struct cublasContext;
using cublasHandle_t = cublasContext*;
using cublasStatus_t = int;
constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;

enum cublasOperation_t { CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C };
enum cublasComputeType_t { CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_16F };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT };
enum cublasMath_t { CUBLAS_DEFAULT_MATH, CUBLAS_TF32_TENSOR_OP_MATH };

inline const char* cublasGetStatusName(cublasStatus_t) {
  return "CUBLAS_STATUS_SUCCESS";
}
inline const char* cublasGetStatusString(cublasStatus_t) { return "no error"; }
inline cublasStatus_t cublasCreate(cublasHandle_t*) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t) {
  return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasGemmEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype,
    int lda, const void* B, cudaDataType Btype, int ldb, const void* beta,
    void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType,
    cublasGemmAlgo_t algo) {
  return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype,
    int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb,
    long long int strideB, const void* beta, void* C, cudaDataType Ctype,
    int ldc, long long int strideC, int batchCount,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  return CUBLAS_STATUS_SUCCESS;
}
