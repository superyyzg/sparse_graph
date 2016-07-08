#include "compute_kernel.cuh"

/*
 * Device code
 */

template< typename T>
__global__ void kernel_soft_threshold( T* dst, T* src, T *thres, uint nd)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (i < nd)
	{
		T st = abs(src[i]) - (*thres);

		T src_sgn = (T)((src[i] > (T)0) - ((src[i] < (T)0)));
		
		if (st < (T)0)
			st = (T)0;

		dst[i]  = st*src_sgn;		
	}
}

template< typename T>
void __global__ kernel_obj_mat(T *d_obj_mat, const T* d_all_sols, const T* d_alphai_prox, const T* d_alpha_neighbor, const T* d_W0_neighbor, const T *d_c, const T *d_lambda_l1, const T *d_lambda_l0, 
							   uint nd, uint nn)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	T c = *d_c;
	T lambda_l1 = *d_lambda_l1;
	T lambda_l0 = *d_lambda_l0;

	if (i < nd && j < (nn+1))
	{
		uint idx = j*nd+i;
		T val = c/2.0*(d_all_sols[idx] - d_alphai_prox[i])*(d_all_sols[idx] - d_alphai_prox[i]) + lambda_l1*abs(d_all_sols[idx]);
		for (uint t = 0; t < nn; t++)
		{
			uint alpha_neighbor_idx = t*nd+i;
			T l0_val = (T)(abs(d_all_sols[idx] -d_alpha_neighbor[alpha_neighbor_idx]) > eps);
			val = val + lambda_l0*d_W0_neighbor[t]*l0_val;
		}
		d_obj_mat[idx] = val;
	}

}

template< typename T>
__global__ void kernel_alphai_from_obj_mat( T* d_alphai, const T* d_obj_mat, const T* d_all_sols, uint nd, uint nn)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

	uint idx = 0;
	
	if (i < nd)
	{
		T row_min = d_obj_mat[i];
		uint min_col = 0;
		for (int j = 1; j < nn+1; j++)
		{
			idx = j*nd+i;
			if (d_obj_mat[idx] < row_min)
			{
				row_min = d_obj_mat[idx];
				min_col = j;
			}
		}
		d_alphai[i] = d_all_sols[min_col*nd+i];
	}
}

template< typename T>
__global__ void kernel_set_alphi(T* d_alphai, uint data_i)
{
	d_alphai[data_i] = 0;
}


/*
function [alphai,obj_mat] = sol_l0_l1(alphai_prox,alpha_neighbor,W0_neighbor,c,lambda_l1,lambda_l0)
    
    n = size(alphai_prox,1);
    nn = size(alpha_neighbor,2);
        
    l1_solution = max(abs(alphai_prox) - lambda_l1/c,0).*sign(alphai_prox);
    all_sols = [l1_solution alpha_neighbor];
    
    obj_mat = c/2*(all_sols - repmat(alphai_prox,1,nn+1)).^2 + lambda_l1*(abs(all_sols));
    
    for t = 1:nn,
        obj_mat = obj_mat + lambda_l0*W0_neighbor(t)*(abs(all_sols - repmat(alpha_neighbor(:,t),1,nn+1))>eps);
    end
    
    [~,min_idx] = min(obj_mat,[],2);
    
    alphai = all_sols(sub2ind([n nn+1],(1:n)',min_idx));
end
*/

__forceinline__ void compute_sol_l0_l1(float* d_alphai, float* d_alphai_prox, float* d_alpha_neighbor, float* d_W0_neighbor, uint data_i, uint nd, uint nn, float *d_c, float *d_lambda_l1, float *d_lambda_l1_c,
									   float *d_lambda_l0)
{
	float* d_l1_solution = NULL, *d_all_sols = NULL, *d_obj_mat = NULL;
	cudaMalloc((void**)&d_l1_solution, sizeof(float)*nd);
	cudaMalloc((void**)&d_all_sols, sizeof(float)*nd*(nn+1));
	cudaMalloc((void**)&d_obj_mat, sizeof(float)*nd*(nn+1));
	
	dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
	dim3 blocksPerGrid( (nd+threadsPerBlock.x-1)/threadsPerBlock.x);
	
	kernel_soft_threshold<<<blocksPerGrid, threadsPerBlock>>>(d_l1_solution, d_alphai_prox, d_lambda_l1_c, nd);

	cudaMemcpy(d_all_sols,d_l1_solution,sizeof(float)*nd,cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_all_sols+nd,d_alpha_neighbor,sizeof(float)*nd*nn,cudaMemcpyDeviceToDevice);

	threadsPerBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
	blocksPerGrid = dim3((nd+threadsPerBlock.x-1)/threadsPerBlock.x,(nn+1+threadsPerBlock.y-1)/threadsPerBlock.y);
	kernel_obj_mat<<<blocksPerGrid, threadsPerBlock>>>(d_obj_mat, d_all_sols, d_alphai_prox, d_alpha_neighbor, d_W0_neighbor, d_c, d_lambda_l1, d_lambda_l0, nd, nn);

	kernel_alphai_from_obj_mat<<<blocksPerGrid, threadsPerBlock>>>(d_alphai, d_obj_mat, d_all_sols, nd, nn);

	kernel_set_alphi<<<1, 1>>>(d_alphai, data_i);

	cudaFree(d_l1_solution);
	cudaFree(d_all_sols);
	cudaFree(d_obj_mat);
}