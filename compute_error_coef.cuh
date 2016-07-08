#include "compute_kernel.cuh"

/*
 * Device code
 */

__forceinline__ void compute_error_coef(float *err, float *d_alphai, float *d_alphai0, uint nd, cublasHandle_t& handle)
{
	float *alpha_diff = NULL;
	cudaMalloc((void**)&alpha_diff, sizeof(float)*nd);

	dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
	dim3 blocksPerGrid( (nd+threadsPerBlock.x-1)/threadsPerBlock.x);
	kernel_subtract<<<blocksPerGrid, threadsPerBlock>>>(alpha_diff, d_alphai, d_alphai0, nd);

	cublasStatus_t status = cublasSasum(handle, nd, alpha_diff, 1, err);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	cudaFree(alpha_diff);
}