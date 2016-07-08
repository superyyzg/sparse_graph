#include "compute_kernel.cuh"
#include "compute_mat_product.cuh"
/*
 * Device code
 */

template< typename T>
__global__ void kernel_alpha_proximal( T* d_alpha_proximal, const T* d_alphai, const T* d_df,  const T* d_c, uint nd)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (i < nd)
	{
		d_alpha_proximal[i] = d_alphai[i] - 1.0/(*d_c)*d_df[i];
	}
}

// d_adjmat: size nn x n
__forceinline__ void compute_alpha_proximal(float *d_alpha_proximal, float *d_AtA, float *d_AtX, float *d_alphai, float *d_c,  uint data_i, uint nd, uint nn, cublasHandle_t& handle)
{
	float* d_df = NULL;
	cudaMalloc((void**)&d_df, sizeof(float)*nd);

	cudaMemcpy(d_df,d_AtX+data_i*nd,sizeof(float)*nd,cudaMemcpyDeviceToDevice);

	/*float ta=2.0f; float *alpha = &ta; float tb=-2.0f; float *beta = &tb;
	cublasStatus_t status = cublasSgemv(handle,
				CUBLAS_OP_N, 
				nd,           //rows of matrix A
				nd,           //cols of matrix A
				alpha,       //alpha
				d_AtA,   // A address
				nd,        // lda
				d_alphai,   // x address
				1,
				beta, 
				d_df, 1);

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}*/

	compute_df(d_df, d_AtA, d_alphai, nd, handle);

	
	dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
	dim3 blocksPerGrid((nd+threadsPerBlock.x-1)/threadsPerBlock.x);

	kernel_alpha_proximal<<<blocksPerGrid, threadsPerBlock>>>(d_alpha_proximal, d_alphai, d_df, d_c, nd);

	cudaFree(d_df);
}