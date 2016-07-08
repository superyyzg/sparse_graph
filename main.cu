//#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#include "compute_kernel.cuh"
#include "compute_obj_robust.cuh"
#include "compute_sol_l0_l1.cuh"
#include "compute_alphai_neighbor.cuh"
#include "compute_mat_product.cuh"
#include "compute_alpha_proximal.cuh"
#include "compute_error_coef.cuh"
#include "utility.h"





/*
for outer_iter = 1:maxIter,


for i = 1:n,
    
    alpha_neighbor = alpha(:,adjmat(:,i));
    W0_neighbor = ones(1,knn);
    
	alphai0 = alpha(:,i); 
    alphai = alphai0;
    
    %[obj,l2err,l1_spar_err,l0_spar_err] = compute_obj_robust(X,i,A,alphai0,alpha_neighbor,W0_neighbor,lambda_l1,lambda_l0);
    
    %fprintf('obj is %.5f, l2err is %.5f, l1_spar_err is %.5f l0_spar_err is %.5f\n', obj,l2err,l1_spar_err,l0_spar_err);
    
    iter = 1;
    while ( iter <= maxSingleIter )
        %add robustness to noise and outlier
        df = 2*(AtA*alphai-AtX(:,i));
        c = 2*S(1);
        
        alpha_proximal = alphai - 1/c*df;

        [alphai,~] = sol_l0_l1(alpha_proximal,alpha_neighbor,W0_neighbor,c,lambda_l1,lambda_l0);

        alphai(i) = 0;
        
		err = errorCoef(alphai,alphai0);
		
		alphai0 = alphai;

        [obj,l2err,l1_spar_err,l0_spar_err] = compute_obj_robust(X,i,A,alphai,alpha_neighbor,W0_neighbor,lambda_l1,lambda_l0);
        
        if verbose,
            fprintf('proximal_manifold: errors = [%1.1e], iter: %4.0f \n',err,iter);
            fprintf('obj is %.5f, l2err is %.5f, l1_spar_err is %.5f l0_spar_err is %.5f\n', obj,l2err,l1_spar_err,l0_spar_err);
        end

        iter = iter+1;

    end %while
    
    alpha(:,i) = alphai;
    lastprintlength = textprogressbar(0,lastprintlength,i,n);

end %for

end %outer_iter
*/

int main(int argc, char *argv[])
{
	cudaSetDevice(0);
	
	float lambda_l1 = static_cast<float>(atof(argv[1]));
	float lambda_l0 = static_cast<float>(atof(argv[2]));	
	int maxSingleIter = atoi(argv[3]);
	int maxIter = atoi(argv[4]);

	bool verbose = false;
	if (argc == 6)
		verbose = static_cast<bool>(atoi(argv[1]));		

	//float lambda_l1 = 0.1f, lambda_l0 = 0.1f;
	//int maxSingleIter = 30, maxIter = 5;

	MATFile *l0l1graph_input = matOpen("l0l1graph_input.mat","r");
	mxArray *XArray = matGetVariable(l0l1graph_input, "X");
	mxArray *l1graph_alphaArray = matGetVariable(l0l1graph_input, "l1graph_alpha");
	mxArray *adjmatArray = matGetVariable(l0l1graph_input, "adjmat");
	mxArray *AArray = matGetVariable(l0l1graph_input, "A");
	mxArray *AtAArray = matGetVariable(l0l1graph_input, "AtA");
	mxArray *AtXArray = matGetVariable(l0l1graph_input, "AtX");
	mxArray *S1Array = matGetVariable(l0l1graph_input, "S1");
	mxArray *knnArray = matGetVariable(l0l1graph_input, "knn");

	float *h_X = static_cast<float*>(mxGetData(XArray));
	float *h_l1graph_alpha = static_cast<float*>( mxGetData(l1graph_alphaArray));
	float *h_adjmat = static_cast<float*>(mxGetData(adjmatArray));
	float *h_A = static_cast<float*>( mxGetData(AArray));
	float *h_AtA = static_cast<float*>( mxGetData(AtAArray));
	float *h_AtX = static_cast<float*>( mxGetData(AtXArray));
	float S1 = *(static_cast<float*>( mxGetData(S1Array)));
	uint nn = (uint)(*(static_cast<float*>( mxGetData(knnArray))));

	const mwSize *Xsize = mxGetDimensions(XArray);
	uint d = static_cast<uint>(Xsize[0]);
	uint n = static_cast<uint>(Xsize[1]);
	uint nd = n + d;

	//l0l1IO::process_input(h_X, h_l1graph_alpha, h_adjmat, h_A, h_AtA, S1, n, d, nn, "l0l1graph_input");
	
	float *d_X = NULL, *d_alpha = NULL, *d_adjmat = NULL, *d_A = NULL, *d_AtA = NULL, *d_AtX = NULL;
	float *d_alphai_neighbor = NULL, *d_W0_neighbor = NULL;
	float *d_alphai = NULL, *d_alphai0 = NULL, *d_alphai_proximal = NULL;
	float *d_c = NULL, *d_lambda_l1 = NULL, *d_lambda_l1_c = NULL, *d_lambda_l0 = NULL;

	cudaMalloc((void**)&d_X,				sizeof(float)*d*n);
	cudaMalloc((void**)&d_alpha,			sizeof(float)*(nd)*n);
	cudaMalloc((void**)&d_adjmat,			sizeof(float)*nn*n);
	cudaMalloc((void**)&d_A,				sizeof(float)*d*(nd));
	cudaMalloc((void**)&d_AtA,				sizeof(float)*(nd)*(nd));
	cudaMalloc((void**)&d_AtX,				sizeof(float)*(nd)*n);
	cudaMalloc((void**)&d_alphai_neighbor,	sizeof(float)*(nd)*nn);
	cudaMalloc((void**)&d_W0_neighbor,		sizeof(float)*nn);	
	cudaMalloc((void**)&d_alphai,			sizeof(float)*(nd));
	cudaMalloc((void**)&d_alphai0,			sizeof(float)*(nd));
	cudaMalloc((void**)&d_alphai_proximal,	sizeof(float)*(nd));
	cudaMalloc((void**)&d_c,				sizeof(float));
	cudaMalloc((void**)&d_lambda_l1,		sizeof(float));
	cudaMalloc((void**)&d_lambda_l1_c,		sizeof(float));
	cudaMalloc((void**)&d_lambda_l0,		sizeof(float));

	

	float *h_W0_neighbor = (float*)malloc(sizeof(float)*nn);
	for (uint i = 0; i < nn; i++) h_W0_neighbor[i] = 1.0;
	//debug
	float *h_alphai = (float*)malloc(sizeof(float)*(n+d));
	
	

	cudaMemcpy(d_X,h_X,sizeof(float)*d*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_alpha,h_l1graph_alpha,sizeof(float)*nd*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_adjmat,h_adjmat,sizeof(float)*nn*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_A,h_A,sizeof(float)*d*nd,cudaMemcpyHostToDevice);
	cudaMemcpy(d_AtA,h_AtA,sizeof(float)*nd*nd,cudaMemcpyHostToDevice);
	cudaMemcpy(d_AtX,h_AtX,sizeof(float)*nd*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_W0_neighbor,h_W0_neighbor,sizeof(float)*nn,cudaMemcpyHostToDevice);

	float c = 2.0f*S1, lambda_l1_c = lambda_l1/c;
	cudaMemcpy(d_c,&c,sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_lambda_l1,&lambda_l1,sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_lambda_l1_c,&lambda_l1_c,sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_lambda_l0,&lambda_l0,sizeof(float),cudaMemcpyHostToDevice);

	matClose(l0l1graph_input);


	// create and initialize CUBLAS library object 
	cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS)
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            std::cout << "CUBLAS object instantialization error" << std::endl;
        }
        getchar ();
        return 0;
    }

	float elapsedTime = 0;
	
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);	

	//compute AtX
	//compute_AtX(d_AtX, d_A, d_X, n, d, handle);

	for (int outer_iter = 0; outer_iter < maxIter; outer_iter++)
	{
		for (uint i = 0; i < n; i++)
		{
			//alpha_neighbor = alpha(:,adjmat(:,i));
			//W0_neighbor = ones(1,knn);
			//alphai0 = alpha(:,i); 
			//alphai = alphai0;			

			compute_alphai_neighbor(d_alphai_neighbor, d_adjmat, d_alpha, i, nd, nn);

			cudaMemcpy(d_alphai0,d_alpha+i*nd,sizeof(float)*nd,cudaMemcpyDeviceToDevice);

			cudaMemcpy(d_alphai,d_alphai0,sizeof(float)*nd,cudaMemcpyDeviceToDevice);

			int iter = 0;
			while ( iter < maxSingleIter )
			{
				//df = 2*(AtA*alphai-AtX(:,i));
				//c = 2*S(1);
                //alpha_proximal = alphai - 1/c*df;

				compute_alpha_proximal(d_alphai_proximal, d_AtA, d_AtX, d_alphai, d_c, i, nd, nn, handle);

				//[alphai,~] = sol_l0_l1(alpha_proximal,alpha_neighbor,W0_neighbor,c,lambda_l1,lambda_l0);
				//alphai(i) = 0;
				compute_sol_l0_l1(d_alphai, d_alphai_proximal, d_alphai_neighbor, d_W0_neighbor, i, nd, nn, d_c, d_lambda_l1, d_lambda_l1_c, d_lambda_l0);
				//debug
				//cudaMemcpy(h_alphai,d_alphai,sizeof(float)*nd,cudaMemcpyDeviceToHost);

				//err = errorCoef(alphai,alphai0);
				float err = 0.0f;
				compute_error_coef(&err, d_alphai, d_alphai0, nd, handle);

				//alphai0 = alphai;
				cudaMemcpy(d_alphai0,d_alphai,sizeof(float)*nd,cudaMemcpyDeviceToDevice);

				/*float obj = 0.0f, l2err = 0.0f, l1_spar_err = 0.0f, l0_spar_err = 0.0f;
				compute_obj_robust(obj, l2err, l1_spar_err, l0_spar_err, d_X, i, d_A, d_alphai, d_alphai_neighbor, d_W0_neighbor, lambda_l1, d_lambda_l0, n, nn, d, handle);
				
				//debug
				//cudaMemcpy(h_obj,d_obj,sizeof(float),cudaMemcpyDeviceToHost);
				//cudaMemcpy(h_l2err,d_l2err,sizeof(float),cudaMemcpyDeviceToHost);
				//cudaMemcpy(h_l1_spar_err,d_l1_spar_err,sizeof(float),cudaMemcpyDeviceToHost);
				//cudaMemcpy(h_l0_spar_err,d_l0_spar_err,sizeof(float),cudaMemcpyDeviceToHost);

				if (verbose)
				{
					printf("proximal_manifold: errors = %.5f, iter: %d \n", err, iter);
					printf("obj is %.5f, l2err is %.5f, l1_spar_err is %.5f l0_spar_err is %.5f\n", obj, l2err, l1_spar_err, l0_spar_err);
				}*/

				iter = iter+1;

			}//while

			//alpha(:,i) = alphai;
			cudaMemcpy(d_alpha+i*nd, d_alphai,sizeof(float)*nd,cudaMemcpyDeviceToDevice);

		}//for i = 1:n
		printf("iteration %d finished \n", outer_iter);
	}//for outer_iter

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
	//cudaDeviceSynchronize();
	//finishTime=clock();
	//elapsedTime =(float)(finishTime - startTime);

	// Clean up:
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	printf("time to compute on gpu is %.10f second\n",elapsedTime/(CLOCKS_PER_SEC));




	MATFile *rFile = matOpen("l0l1graph_result.mat","w");	
	
	mxArray* alphaoutArray = mxCreateNumericMatrix(n+d,n, mxSINGLE_CLASS, mxREAL);
	
	float *h_alphaout   = (float*)mxGetData(alphaoutArray);
	
	//transfer the data from gpu to cpu
	cudaMemcpy(h_alphaout,d_alpha,sizeof(float)*(n+d)*n,cudaMemcpyDeviceToHost);

	matPutVariable(rFile, "l0l1graph_alpha", alphaoutArray);
	matClose(rFile);

	mxDestroyArray(alphaoutArray);

	//destroy the input matlab Arrays
	mxDestroyArray(XArray);
	mxDestroyArray(l1graph_alphaArray);
	mxDestroyArray(adjmatArray);
	mxDestroyArray(AArray);
	mxDestroyArray(AtAArray);
	mxDestroyArray(S1Array);
	mxDestroyArray(knnArray);


	//deallocation

	free(h_W0_neighbor);
	//debug
	//free(h_alphai_proximal);

	cudaFree(d_X);
	cudaFree(d_alpha);
	cudaFree(d_adjmat);
	cudaFree(d_A);
	cudaFree(d_AtA);
	cudaFree(d_AtX);
	cudaFree(d_alphai_neighbor);
	cudaFree(d_W0_neighbor);	
	cudaFree(d_alphai);
	cudaFree(d_alphai0);
	cudaFree(d_alphai_proximal);
	cudaFree(d_c);
	cudaFree(d_lambda_l1);
	cudaFree(d_lambda_l1_c);
	cudaFree(d_lambda_l0);

	
	return 0;
}