 #include "common_BS_VA.h"


/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = sqrtf(T);
    d1 = (__logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}

////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void BlackScholesGPU(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
	int iter_per_block
)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    //const int opt = blockDim.x * blockIdx.x + threadIdx.x;

	
	for (int k=0; k<iter_per_block; k++) {
	
		for (int j=0; j< BS_COARSENING; j++) {
		
			const int opt = k * gridDim.x * blockDim.x * BS_COARSENING + j * gridDim.x * blockDim.x +  blockIdx.x * blockDim.x + threadIdx.x;
			
    //No matter how small is execution grid or how large OptN is,
    //exactly OptN indices will be processed with perfect memory coalescing
    //for (int opt = tid; opt < optN; opt += THREAD_N)
			if (opt < optN){
		//for (int i=0;i<10;i++)
				BlackScholesBodyGPU(
					d_CallResult[opt],
					d_PutResult[opt],
					d_StockPrice[opt], 
					d_OptionStrike[opt],
					d_OptionYears[opt],
					Riskfree,
					Volatility
				);
			}
		}
	}
}

/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#include <math.h>



///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
static double CND(double d)
{
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
static void BlackScholesBodyCPU(
    float &callResult,
    float &putResult,
    float Sf, //Stock price
    float Xf, //Option strike
    float Tf, //Option years
    float Rf, //Riskless rate
    float Vf  //Volatility rate
)
{
    double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

    double sqrtT = sqrt(T);
    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double    d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);

    //Calculate Call and Put simultaneously
    double expRT = exp(- R * T);
    callResult   = (float)(S * CNDD1 - X * expRT * CNDD2);
    putResult    = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}


////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options
////////////////////////////////////////////////////////////////////////////////
void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{
    for (int opt = 0; opt < optN; opt++)
        BlackScholesBodyCPU(
            h_CallResult[opt],
            h_PutResult[opt],
            h_StockPrice[opt],
            h_OptionStrike[opt],
            h_OptionYears[opt],
            Riskfree,
            Volatility
        );
}



#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
// extern "C" void BlackScholesCPU(
    // float *h_CallResult,
    // float *h_PutResult,
    // float *h_StockPrice,
    // float *h_OptionStrike,
    // float *h_OptionYears,
    // float Riskfree,
    // float Volatility,
    // int optN
// );



////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
// const int OPT_N = 4000000;
// const int  NUM_ITERATIONS = 512;


// const int          OPT_SZ = OPT_N * sizeof(float);
// const float      RISKFREE = 0.02f;
// const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

/// Global memory //////

//'h_' prefix - CPU (host) memory space
    float
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult=NULL,
    *d_PutResult=NULL,
    //GPU instance of input data
    *d_StockPrice=NULL,
    *d_OptionStrike=NULL,
    *d_OptionYears=NULL;

    double
    delta, ref, sum_delta, sum_ref, max_delta, L1norm;
	
	float	RISKFREE = 0.02f;
	float   VOLATILITY = 0.30f;
	int 	OPT_SZ;
	int OPT_N;

	


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int BS_start_kernel(int TB_Number, int Blk_Size, int iter_per_block)
{
	OPT_N = TB_Number * BS_COARSENING * Blk_Size * iter_per_block;
	
	OPT_SZ = OPT_N * sizeof(float);

    // Start logs

    //printf("Initializing data...\n");
    //printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);

    //printf("...allocating GPU memory for options.\n");
    checkCudaErrors(cudaMalloc((void **)&d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears,  OPT_SZ));

    // printf("...generating input data in CPU mem.\n");
    srand(5347);

    //Generate options set
    for (int i = 0; i < OPT_N; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }

    //printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
    //printf("Data init done.\n\n");

    checkCudaErrors(cudaMemset(d_CallResult, 0, OPT_SZ));
    checkCudaErrors(cudaMemset(d_PutResult, 0, OPT_SZ));


    //printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    checkCudaErrors(cudaDeviceSynchronize());
    
	
	return 0;
	}
	
int BS_execute_kernel (int TB_Number, int Blk_Size, int iter_per_block, float *time)
{
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	
	int NUM_ITERATIONS=512;
    sdkStartTimer(&hTimer);
	//printf("ejecutando BS con NB=%d BS=%d OPT_N=%d iter_per_block=%d\n", TB_Number, Blk_Size, OPT_N, iter_per_block); 
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        BlackScholesGPU<<<TB_Number, Blk_Size/*DIV_UP(OPT_N, 128), 128*/>>>(
            d_CallResult,
            d_PutResult,
            d_StockPrice,
            d_OptionStrike,
            d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N,
			iter_per_block
        );
        //getLastCudaError("BlackScholesGPU() execution failed\n");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    float gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;
	sdkDeleteTimer(&hTimer);
	*time = gpuTime;
	
	return 0;
}
	
int BS_end_kernel()
{
    // printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
    checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));


    //printf("Checking the results...\n");
   // printf("...running CPU calculations.\n\n");
    //Calculate options values on CPU
    BlackScholesCPU(
        h_CallResultCPU,
        h_PutResultCPU,
        h_StockPrice,
        h_OptionStrike,
        h_OptionYears,
        RISKFREE,
        VOLATILITY,
        OPT_N
    );

    //printf("Comparing the results...\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (int i = 0; i < OPT_N; i++)
    {
        ref   = h_CallResultCPU[i];
        delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
   
//   printf("L1 norm: %E\n", L1norm);
//    printf("Max absolute error: %E\n\n", max_delta);

    //printf("Shutting down...\n");
    //printf("...releasing GPU memory.\n");
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));

    //printf("...releasing CPU memory.\n");
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);
    
    //printf("Shutdown done.\n");

    //printf("\n[BlackScholes] - Test Summary\n");
   // cudaDeviceReset();

    if (L1norm > 1e-3)
    {
        printf("Test failed!\n");
        //exit(EXIT_FAILURE);
    }

    //printf("Test passed\n");
	
	return 0;
}
