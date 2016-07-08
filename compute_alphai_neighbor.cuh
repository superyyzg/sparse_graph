#include "compute_kernel.cuh"

/*
 * Device code
 */

template< typename T>
__global__ void kernel_alpha_neighbor( T* d_alphai_neighbor, const T* d_adjmat, const T* d_alpha, uint data_i, uint nd, uint nn)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < nd && j < nn)
	{
		uint idx = j*nd+i;
		d_alphai_neighbor[idx] = d_alpha[(uint)(d_adjmat[data_i*nn+j]-1)*nd + i];
	}
}

// d_adjmat: size nn x n
__forceinline__ void compute_alphai_neighbor(float *d_alphai_neighbor, float *d_adjmat, float *d_alpha, uint data_i, uint nd, uint nn)
{
	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 blocksPerGrid((nd+threadsPerBlock.x-1)/threadsPerBlock.x,(nn+threadsPerBlock.y-1)/threadsPerBlock.y);

	kernel_alpha_neighbor<<<blocksPerGrid, threadsPerBlock>>>(d_alphai_neighbor, d_adjmat, d_alpha, data_i, nd, nn);
}