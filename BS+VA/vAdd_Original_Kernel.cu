/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
#include "common_BS_VA.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

__global__ void
vectorAdd(const float *A, const float *B, float *C, int iter_per_block, int numelements)
{
	int i;
	
	for (int k=0; k<iter_per_block; k++) {
	
		for (int j=0;j<VA_COARSENING;j++) {
			//int vId = blkDim*bid + threadIdx.x;
			
			const int i = k * gridDim.x * blockDim.x * VA_COARSENING + j * gridDim.x * blockDim.x +  
			blockIdx.x * blockDim.x + threadIdx.x;
			
			if (i < numelements)
					C[i] = A[i] + B[i];
		}
	}

}



/**
 * Host main routine
 */
 
//// Global variables

 float *h_A;
 float *h_B;
 float *h_C;
 float *d_A;
 float *d_B;
 float *d_C;
 
 int numElements;
  
int VA_start_kernel(int TB_Number, int Blk_Size, int iter_per_block)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    /*int*/ numElements = TB_Number * VA_COARSENING * Blk_Size * iter_per_block;
    size_t size = numElements * sizeof(float);
    //printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    h_A = (float *)malloc(size);

    // Allocate the host input vector B
    h_B = (float *)malloc(size);

    // Allocate the host output vector C
    h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    checkCudaErrors(cudaMemset(d_C, 0, size));


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	return 0;
}
	
int VA_execute_kernel(int TBN, int Blk_Size, int iter_per_block, float *time)
{
     cudaError_t err = cudaSuccess;
	
	// Launch the Vectoir Add CUDA Kernel
    int threadsPerBlock = Blk_Size;
    int blocksPerGrid = TBN;
	//numElements = blocksPerGrid * threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	
	//printf("Ejecutando VA con NB=%d BS=%d numElements=%d\n", blocksPerGrid,threadsPerBlock, numElements); 
	int NUM_ITERATIONS=512;
	sdkStartTimer(&hTimer);
	for (int i=0; i<NUM_ITERATIONS; i++)
		vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, iter_per_block, numElements);
		
	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	
	err = cudaGetLastError();
	
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	float gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;
	sdkDeleteTimer(&hTimer);
	*time = gpuTime;
	
	return 0;
}

int VA_end_kernel()
{
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    
	 cudaError_t err = cudaSuccess;
	 err = cudaMemcpy(h_C, d_C, numElements*sizeof(float), cudaMemcpyDeviceToHost);

     if (err != cudaSuccess)
     {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
     }

    // // Verify that the result vector is correct
     for (int i = 0; i < numElements; ++i)
     {
         if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
         {
             fprintf(stderr, "Result verification failed at element %d!\n", i);
             exit(EXIT_FAILURE);
         }
     }
    //printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    //err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //printf("Done\n");
    return 0;
}
