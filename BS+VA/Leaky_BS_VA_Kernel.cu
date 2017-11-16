#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
#include "common_BS_VA.h"
#include <math.h>

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

__device__ uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}
 
__device__ void LB_vectorAdd(float* d_A, float* d_B, float* d_C, int size, int bid, 
								int iter_per_block, int blkDim, int gridDim){


	//if (threadIdx.x == 1 && bid == 1)
	//	printf("ipb =%d gridDim=%d blkDim=%d size=%d\n", iter_per_block,  gridDim, blkDim, size);
		
	for (int k=0; k<iter_per_block; k++) {
	
		for (int j=0; j< VA_COARSENING; j++) {
	
			int vId = k * gridDim * blkDim * VA_COARSENING + j * gridDim * blkDim + 
			bid * blkDim + threadIdx.x;
			
			//if ((threadIdx.x < 16) && bid == 1 && k == 0 && j == 0)
			//	printf("Tid=%d vId =%d k=%d\n", threadIdx.x, vId, k);
		 
			if(vId < size)
				d_C[vId] = d_A[vId] + d_B[vId];
		}
	}
	//__syncthreads();
}

// Functions declaration
///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////

__device__ inline float cndGPU(float d);
int VA_start_kernel(int TB_Number, int Blk_Size);
int BS_start_kernel(int TB_Number, int Blk_Size);
int VA_end_kernel();
int BS_end_kernel();
int VA_execute_kernel(int TB_Number, int Blk_Size, float *time);
int BS_execute_kernel(int TB_Number, int Blk_Size, float *time);

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


__device__ void LB_BlackScholesGPU(
	float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,

    //float * __restrict d_CallResult,
    //float * __restrict d_PutResult,
    //float * __restrict d_StockPrice,
    //float * __restrict d_OptionStrike,
    //float * __restrict d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
    int bid,
	int iter_per_block,
    int blkDim,
    int gridDim
)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

        //const int opt = blkDim * bid + threadIdx.x;
		
	for (int k=0; k<iter_per_block; k++) {
	
		for (int j=0; j< BS_COARSENING; j++) {
	
			const int opt = k * gridDim * blkDim * BS_COARSENING + j * gridDim * blkDim + 
				bid * blkDim + threadIdx.x;
		 
			if(opt < optN /*&& threadIdx.x < blkDim*/){

                //const int opt = blkDim * bid + threadIdx.x;

                // Calculating 2 options per thread to increase ILP (instruction level parallelism)
                //if (opt < optN)
				//for (int i=0; i<10; i++)
             
                BlackScholesBodyGPU(
					d_CallResult[opt],
					d_PutResult[opt],
					d_StockPrice[opt],
                    d_OptionStrike[opt],
                    d_OptionYears[opt],
                    Riskfree,
                    Volatility);
            }
        }
	}
		
}

__global__ void scheduler_SMT_ver3(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
    int gridDim_BS,
    int blkDim_BS,
    float* A, float* B, float* C, int size,
    int gridDim_vA, int blkDim_vA,
	int iter_per_blockvA, int iter_per_blockBS,
    int *cont_BS, int *cont_vA, int *Th_smid)
{
	unsigned int SM_id, bid;
	__shared__ int s_bid;

		//if (threadIdx.x == 0 && blockIdx.x == 0) printf("GPU: NUM_BLOCK_BS=%d NUM_BLOCKS_vA=%d Total=%d\n", gridDim_BS, gridDim_vA, gridDim.x);

		SM_id = get_smid();
		// Increase the number of blocks executed for each kernel
		if (SM_id < *Th_smid){
			if (threadIdx.x == 0) {
				bid = atomicAdd(cont_BS, 1);
				if (bid < gridDim_BS) {
					s_bid = bid << 1; // Less significant bit to 0 indicates block execution of BS
					//printf("BS_l SMID=%d Blq=%d bid=%d\n", SM_id, gridDim.x, bid);
				}
				else {
					bid = atomicAdd(cont_vA, 1);
					s_bid = (bid << 1) + 1; // Less significant bit to 1 indicates block execution of vAa
					//printf("vA_2  SMID=%d Blq=%d bid=%d\n", SM_id, gridDim.x, bid);
				}
				
			}
		}
		else {
			if (threadIdx.x == 0){
				bid = atomicAdd(cont_vA, 1);
				if (bid < gridDim_vA){
					s_bid = (bid << 1) + 1;
					//printf("vA_l SMID=%d Blq=%d bid=%d\n", SM_id, gridDim.x, bid);
				}			
				else {
					bid = atomicAdd(cont_BS, 1);
					s_bid = bid << 1;
					//printf("BS_2 SMID=%d Blq=%d bid=%d\n", SM_id, gridDim.x, bid);
				}
			}
		}

		__syncthreads();

		/*if (s_bid & 0x00001 == 1 ){
			if (threadIdx.x  == 0)
				printf("vA SMid=%2d bid=%4d rid=%4d \n", SM_id, s_bid>>1, blockIdx.x);
		}
		else
			if (threadIdx.x == 0)
				printf("BS SMid=%2d bid=%4d rid=%4d \n", SM_id, s_bid>>1, blockIdx.x);

		*/

		if (s_bid & 0x00001 == 1 )
				LB_vectorAdd(A, B, C, size, s_bid >> 1, iter_per_blockvA, blkDim_vA, gridDim_vA);
		else
            LB_BlackScholesGPU(
                            (float *)d_CallResult,
                            (float *)d_PutResult,
                            (float *)d_StockPrice,
                            (float *)d_OptionStrike,
                            (float *)d_OptionYears,
                            Riskfree,
                            Volatility,
                            optN,
                            s_bid >> 1,
							iter_per_blockBS,
                            blkDim_BS,
                            gridDim_BS
                        );

}



__global__ void scheduler_SMT_ver2(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
    int gridDim_BS,
    int blkDim_BS,
    float* A, float* B, float* C, int size,
    int gridDim_vA, int blkDim_vA,
	int iter_per_blockvA, int iter_per_blockBS,
    int *cont_BS, int *cont_vA, int *Th_smid)
{
	unsigned int SM_id, bid;
	bool flag=0;
	__shared__ int s_bid;

		//if (threadIdx.x == 0 && blockIdx.x == 0) printf("GPU: NUM_BLOCK_BS=%d NUM_BLOCKS_vA=%d Total=%d\n", gridDim_BS, gridDim_vA, gridDim.x);

		SM_id = get_smid();

		if (threadIdx.x == 0) {
		// Increase the number of blocks executed for each kernel
			if (SM_id < *Th_smid){
				bid = *cont_BS;
				while (bid < gridDim_BS){
					if (atomicCAS(cont_BS, bid, bid+1) == bid){
						flag = 1;
						break;
					}
					bid = *cont_BS;
				}

				if (flag == 0){
					bid = atomicAdd(cont_vA, 1);
					s_bid = (bid << 1) + 1; // Less significant bit to 1 indicates bllock execution of vA
				}
				else {
					s_bid = bid << 1; // Less significant bit to 0 indicates block execution of BS
				}
			}
			else {
				bid = *cont_vA;
				while (bid < gridDim_vA){
					if (atomicCAS(cont_vA, bid, bid+1) == bid){
						flag = 1;
						break;
					}
					bid = *cont_vA;
				}

				if (flag == 0){
					bid = atomicAdd(cont_BS, 1);
					s_bid = bid << 1; // Less significant bit to 0 indicates block execution of BS

				}
				else {
					s_bid = (bid << 1) + 1; // Less significant bit to 1 indicates block execution of vA
				}
			}
		}

		__syncthreads();


		if (s_bid & 0x00001 == 1 ){
				LB_vectorAdd(A, B, C, size, s_bid >> 1, iter_per_blockvA, blkDim_vA, gridDim_vA);
		}
		else
            LB_BlackScholesGPU(
                            (float *)d_CallResult,
                            (float *)d_PutResult,
                            (float *)d_StockPrice,
                            (float *)d_OptionStrike,
                            (float *)d_OptionYears,
                            Riskfree,
                            Volatility,
                            optN,
                            s_bid >> 1,
							iter_per_blockBS,
                            blkDim_BS,
                            gridDim_BS
                        );

}


__global__ void scheduler_SMT(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
    int gridDim_BS,
    int blkDim_BS,
    float* A, float* B, float* C, int size,
    int gridDim_vA, int blkDim_vA,
	int iter_per_blockvA, int iter_per_blockBS,
    int *cont_BS, int *cont_vA, int *Th_smid)
{
	unsigned int SM_id, bid;
	__shared__ int s_bid;
	
		//if (threadIdx.x == 0 && blockIdx.x == 0) printf("GPU: NUM_BLOCK_BS=%d NUM_BLOCKS_vA=%d Total=%d\n", gridDim_BS, gridDim_vA, gridDim.x);
		
		SM_id = get_smid();
		// Increase the number of blocks executed for each kernel
		if (SM_id < *Th_smid){
			if (*cont_BS <gridDim_BS){
				if (threadIdx.x == 0) {
					bid = atomicAdd(cont_BS, 1);
					s_bid = bid << 1; // Less significant bit to 0 indicates block execution of BS 
					//printf("1-BS SMid=%2d bid=%4d rid=%4d \n", SM_id, bid, blockIdx.x);
				}
			}
			else{
				if (threadIdx.x == 0){
						bid = atomicAdd(cont_vA, 1);
						s_bid = (bid << 1) + 1; // Less significant bit to 1 indicates block execution of vA
						//printf("1-vA SMid=%2d bid=%4d\n", SM_id, bid);
				}
			}
		}	
		else {
			if (*cont_vA < gridDim_vA){
				if (threadIdx.x == 0) {
					bid = atomicAdd(cont_vA, 1);
					s_bid = (bid << 1) + 1;
					//printf("2-vA SMid=%2d bid=%4d\n", SM_id, bid);
				}
			}
			else {
				if (threadIdx.x == 0) {
					bid = atomicAdd(cont_BS, 1); 
					s_bid = bid << 1;
				//	printf("2-BS SMid=%2d bid=%4d rid=%4d \n", SM_id, bid, blockIdx.x);
				}
			}
		}
				
		
		__syncthreads();
		
		/*if (s_bid & 0x00001 == 1 ){
			if (threadIdx.x  == 0)
				printf("vA SMid=%2d bid=%4d rid=%4d \n", SM_id, s_bid>>1, blockIdx.x);
		}
		else
			if (threadIdx.x == 0)
				printf("BS SMid=%2d bid=%4d rid=%4d \n", SM_id, s_bid>>1, blockIdx.x);
		
		*/
		
		if (s_bid & 0x00001 == 1 )
				LB_vectorAdd(A, B, C, size, s_bid >> 1, iter_per_blockvA, blkDim_vA, gridDim_vA);
		else
            LB_BlackScholesGPU(
                            (float *)d_CallResult,
                            (float *)d_PutResult,
                            (float *)d_StockPrice,
                            (float *)d_OptionStrike,
                            (float *)d_OptionYears,
                            Riskfree,
                            Volatility,
                            optN,
                            s_bid >> 1,
							iter_per_blockBS,
                            blkDim_BS,
                            gridDim_BS
                        );

}


__global__ void scheduler_SMK(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
    int gridDim_BS,
    int blkDim_BS,
    float* A, float* B, float* C, int size,
    int gridDim_vA, int blkDim_vA,
	int iter_per_blockvA, int iter_per_blockBS,
	int num_blockvA_perSM, int num_blockBS_perSM, 
    int *cont_SMs, int *SM_blocks_layout, int *block_index)
{
	int SM_id;
	__shared__ int bid;
	
	//	if (threadIdx.x == 0 && blockIdx.x == 0) printf("GPU: NUM_BLOCK_BS=%d NUM_BLOCKS_vA=%d Total=%d\n", gridDim_BS, gridDim_vA, gridDim.x);
		int max_block_per_SM = num_blockvA_perSM + num_blockBS_perSM;
		
		SM_id = get_smid();
		
		// Assign an index to the blocks running at each SM
		if (threadIdx.x == 0)
			bid = atomicAdd(&cont_SMs[SM_id], 1);
			
		__syncthreads();
		
		if (SM_blocks_layout[bid] == 0)
				LB_vectorAdd(A, B, C, size, block_index[bid]+SM_id * num_blockvA_perSM, iter_per_blockvA, 
							blkDim_vA, gridDim_vA);
		else
            LB_BlackScholesGPU(
                            (float *)d_CallResult,
                            (float *)d_PutResult,
                            (float *)d_StockPrice,
                            (float *)d_OptionStrike,
                            (float *)d_OptionYears,
                            Riskfree,
                            Volatility,
                            optN,
                            block_index[bid]+SM_id * num_blockBS_perSM,
							iter_per_blockBS,
                            blkDim_BS,
                            gridDim_BS
                        );

}

__global__ void scheduler_SMK_ver2(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
    int gridDim_BS,
    int blkDim_BS,
    float* A, float* B, float* C, int size,
    int gridDim_vA, int blkDim_vA,
	int iter_per_blockvA, int iter_per_blockBS,
	int num_blockvA_perSM, int num_blockBS_perSM, 
    int *cont_SMs, int *SM_blocks_layout,
	int *cont_BS, int* cont_vA)
{
	int SM_id, bid;
	__shared__ int s_bid;
	
	//	if (threadIdx.x == 0 && blockIdx.x == 0) printf("GPU: NUM_BLOCK_BS=%d NUM_BLOCKS_vA=%d Total=%d\n", gridDim_BS, gridDim_vA, gridDim.x);
		int max_block_per_SM = num_blockvA_perSM + num_blockBS_perSM;

		SM_id = get_smid();
		
		// Assign an index to the blocks running at each SM
		if (threadIdx.x == 0) {
			const int index = atomicAdd(&cont_SMs[SM_id], 1);
			if ( SM_blocks_layout[index % max_block_per_SM] == 0) {
				if (*cont_BS <gridDim_BS-1){  
					bid = atomicAdd(cont_BS, 1);
					s_bid = bid << 1; // Less significant bit to 0 indicates block execution of BS 
						//printf("BS SMid=%2d bid=%4d rid=%4d \n", SM_id, bid, blockIdx.x);
				}
				else {
					bid = atomicAdd(cont_vA, 1);
					s_bid = (bid << 1) + 1;
				}
			}
			else {
				if (*cont_vA <gridDim_vA-1){
					bid = atomicAdd(cont_vA, 1);
					s_bid = (bid  << 1) + 1 ;
				}
				else{
					bid = atomicAdd(cont_BS, 1);
					s_bid = bid << 1;
				}
			}
		}
			
		__syncthreads();
		
		if ( s_bid & 0x1 == 1)
		
			LB_vectorAdd(A, B, C, size, s_bid>>1, iter_per_blockvA, 
							blkDim_vA, gridDim_vA);
		else
				LB_BlackScholesGPU(
                            (float *)d_CallResult,
                            (float *)d_PutResult,
                            (float *)d_StockPrice,
                            (float *)d_OptionStrike,
                            (float *)d_OptionYears,
                            Riskfree,
                            Volatility,
                            optN,
                            s_bid>>1,
							iter_per_blockBS,
                            blkDim_BS,
                            gridDim_BS
                        );

}

__global__ void scheduler_SMK_ver3(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
    int gridDim_BS,
    int blkDim_BS,
    float* A, float* B, float* C, int size,
    int gridDim_vA, int blkDim_vA,
	int iter_per_blockvA, int iter_per_blockBS,
	int num_blockvA_perSM, int num_blockBS_perSM,
    int *cont_SMs, int *SM_blocks_layout,
	int *cont_BS, int* cont_vA)
{
	int SM_id, bid;
	__shared__ int s_bid;

	//	if (threadIdx.x == 0 && blockIdx.x == 0) printf("GPU: NUM_BLOCK_BS=%d NUM_BLOCKS_vA=%d Total=%d\n", gridDim_BS, gridDim_vA, gridDim.x);
		int max_block_per_SM = num_blockvA_perSM + num_blockBS_perSM;

		SM_id = get_smid();

		// Assign an index to the blocks running at each SM
		if (threadIdx.x == 0) {
			const int index = atomicAdd(&cont_SMs[SM_id], 1);
			if ( SM_blocks_layout[index % max_block_per_SM] == 0) {
				bid = atomicAdd(cont_BS, 1);
				if (bid <gridDim_BS){
					s_bid = bid << 1; // Less significant bit to 0 indicates block execution of BS
						//printf("1- BS SMid=%2d bid=%4d rid=%4d \n", SM_id, bid, blockIdx.x);
				}
				else {
					bid = atomicAdd(cont_vA, 1);
					s_bid = (bid << 1) + 1;
					//printf("2- VA SMid=%2d bid=%4d rid=%4d \n", SM_id, bid, blockIdx.x);

				}
			}
			else {
				bid = atomicAdd(cont_vA, 1);
				if (bid <gridDim_vA) {
					s_bid = (bid  << 1) + 1 ;
					//printf("1- VA SMid=%2d bid=%4d rid=%4d \n", SM_id, bid, blockIdx.x);
				}
				else{
					bid = atomicAdd(cont_BS, 1);
					s_bid = bid << 1;
					//printf("2- BS SMid=%2d bid=%4d rid=%4d \n", SM_id, bid, blockIdx.x);
				}
			}
		}

		__syncthreads();

		if ( s_bid & 0x1 == 1)

			LB_vectorAdd(A, B, C, size, s_bid>>1, iter_per_blockvA,
							blkDim_vA, gridDim_vA);
		else
				LB_BlackScholesGPU(
                            (float *)d_CallResult,
                            (float *)d_PutResult,
                            (float *)d_StockPrice,
                            (float *)d_OptionStrike,
                            (float *)d_OptionYears,
                            Riskfree,
                            Volatility,
                            optN,
                            s_bid>>1,
							iter_per_blockBS,
                            blkDim_BS,
                            gridDim_BS
                        );

}



extern float * d_CallResult;
extern float * d_PutResult;
extern float * d_StockPrice;
extern float * d_OptionStrike;
extern float *  d_OptionYears;
extern float RISKFREE;
extern float VOLATILITY;
extern int OPT_N;

extern float *d_A;
extern float *d_B;
extern float *d_C;
extern int numElements;


int Leaky_SMT(int TB_BS, int TB_vA, int blkDim, float *time, int iter_per_blockvA, int iter_per_blockBS, int th_simd, int num_SMs)
{

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	
	/** Prepare both kernels ***/
		
	if (TB_BS>0) BS_start_kernel(TB_BS, blkDim, iter_per_blockBS);
	if (TB_vA>0) VA_start_kernel(TB_vA, blkDim, iter_per_blockvA);
		
	//printf("Execution SMT -- TB_BS=%d TB_VA=%d I_BS=%d I_VA=%d size_BS=%d size_VA=%d\n", 
	//TB_BS, TB_vA, iter_per_blockBS, iter_per_blockvA, OPT_N, numElements);
	
	int blocks = TB_BS + TB_vA;
		
	int NUM_ITERATIONS=512;

	int *d_th_simd; //It is going to store the number of SMs devoted to BS kernels computation
	checkCudaErrors(cudaMalloc((void**)&d_th_simd, sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_th_simd, &th_simd, sizeof(int), cudaMemcpyHostToDevice)); 

	int *d_cont_BS;
	checkCudaErrors(cudaMalloc((void**)&d_cont_BS, sizeof(int)));
		
	int *d_cont_vA;
	checkCudaErrors(cudaMalloc((void**)&d_cont_vA, sizeof(int)));
		
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
		
	for (int ii=0 ;ii<NUM_ITERATIONS; ii++) {
		
		checkCudaErrors(cudaMemset(d_cont_BS, 0, sizeof(int)));
		checkCudaErrors(cudaMemset(d_cont_vA, 0, sizeof(int)));
		
		
		scheduler_SMT_ver3<<<blocks, blkDim>>>(
    				
    	            (float *)d_CallResult,
    	            (float *)d_PutResult,
    	            (float *)d_StockPrice,
    	            (float *)d_OptionStrike,
    	            (float *)d_OptionYears,
    	            RISKFREE,
    	            VOLATILITY,
    	            OPT_N,
    	            TB_BS, blkDim, //liczba blok�w watk�w i watk�w na blok BlackScholes
    	            d_A, d_B, d_C, numElements,
    	            
    	            TB_vA, blkDim, //liczba blok�w watk�w i watk�w na blok vectorAdd
					iter_per_blockvA, iter_per_blockBS,
    	            
    	            d_cont_BS, d_cont_vA, d_th_simd
			);
	} 



	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	*time = sdkGetTimerValue(&hTimer) / (float)NUM_ITERATIONS;
	
	// Check block counters

	int h_cont_BS, h_cont_vA;
    checkCudaErrors(cudaMemcpy(&h_cont_BS, d_cont_BS, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_cont_vA,  d_cont_vA, sizeof(int), cudaMemcpyDeviceToHost));

   /* if (h_cont_BS != TB_BS || h_cont_vA != TB_vA){
    	printf("!! Error: Some blocks have not been executed BS=%d vA=%d\n", h_cont_BS, h_cont_vA);
    }
*/


	if (TB_BS>0) BS_end_kernel();
	if (TB_vA>0) VA_end_kernel();
	
	checkCudaErrors(cudaFree(d_cont_BS));
	checkCudaErrors(cudaFree(d_cont_vA));
	checkCudaErrors(cudaFree(d_th_simd));
	
	return 0;
	
}


int Leaky_SMK(int TB_BS, int TB_vA, int blkDim, float *time, int iter_per_blockvA, int iter_per_blockBS,
	int num_blockvA_perSM, int num_blockBS_perSM, int num_SMs)
{

	float time_par, time_acc=0;
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
/* Run BS only with Leaky bucket code. Thus Leaky bucket overhead can be calculated*/
	
	/** Prepare both kernels ***/
		
		if (TB_BS >0)
			BS_start_kernel(TB_BS, blkDim, iter_per_blockBS);
		if (TB_vA >0)
			VA_start_kernel(TB_vA, blkDim, iter_per_blockvA);
		
		//printf("Execution SMK -- TB_BS=%d TB_VA=%d I_BS=%d I_VA=%d size_BS=%d size_VA=%d blkDim=%d\n",
		//	TB_BS, TB_vA, iter_per_blockBS, iter_per_blockvA, OPT_N, numElements, blkDim);
		
		// Generate cpu arrays 
		
		int *SM_blocks_layout, *block_index;
		
		if ((SM_blocks_layout = (int *)malloc((num_blockvA_perSM + num_blockBS_perSM) *sizeof(int)))==NULL){			
			printf("SM_blocks_layout cannot be created\n");
			exit(-1);
		}
		
		if ((block_index = (int *)malloc((num_blockvA_perSM + num_blockBS_perSM) *sizeof(int)))==NULL){			
			printf("block_index cannot be created\n");
			exit(-1);
		}
		
		int num_blockvA, num_blockBS;
		
		num_blockvA = num_blockvA_perSM;
		num_blockBS = num_blockBS_perSM;
		int cont_vA=0, cont_BS=0;
		
		for (int i=0; i<num_blockvA_perSM + num_blockBS_perSM; i++) {
			
			if (( i % 2) == 0){
				if (num_blockvA>0){
					SM_blocks_layout[i]=0;
					block_index[i]=cont_vA;
					num_blockvA--;
					cont_vA++;
				}
				else {
					SM_blocks_layout[i]=1;
					block_index[i]=cont_BS;
					num_blockBS--;
					cont_BS++;
				}
			}
			
			if (( i % 2) == 1){
				if (num_blockBS>0){
					SM_blocks_layout[i]=1;
					block_index[i]=cont_BS;
					num_blockBS--;
					cont_BS++;
				}
				else {
					SM_blocks_layout[i]=0;
					block_index[i]=cont_vA;
					num_blockvA--;
					cont_vA++;
				}
			}
		}
				
		// Copy maps to GPU memory
		
    	int* d_SM_blocks_layout;
    	checkCudaErrors(cudaMalloc((void**)&d_SM_blocks_layout, sizeof(int)*(num_blockvA_perSM + num_blockBS_perSM)));
    	checkCudaErrors(cudaMemcpy(d_SM_blocks_layout, SM_blocks_layout, sizeof(int)*(num_blockvA_perSM + num_blockBS_perSM), cudaMemcpyHostToDevice));
		
		int *d_cont_SMs;
		checkCudaErrors(cudaMalloc((void**)&d_cont_SMs, sizeof(int)*num_SMs));
		
		int *d_cont_BS;
		checkCudaErrors(cudaMalloc((void**)&d_cont_BS, sizeof(int)));
		
		int *d_cont_vA;
		checkCudaErrors(cudaMalloc((void**)&d_cont_vA, sizeof(int)));
		
		sdkResetTimer(&hTimer);
		
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
 
    	/*URUCHOMIENIE KERNELA SCHEDULERA*/
		int NUM_ITERATIONS=512;
		sdkStartTimer(&hTimer);
		
		for (int ii=0 ;ii<NUM_ITERATIONS; ii++) {
		
			checkCudaErrors(cudaMemset(d_cont_SMs, 0, sizeof(int)*num_SMs));
			checkCudaErrors(cudaMemset(d_cont_BS, 0, sizeof(int)));
			checkCudaErrors(cudaMemset(d_cont_vA, 0, sizeof(int)));

			
			cudaEventRecord(start, 0);
			
			scheduler_SMK_ver3<<<TB_BS+TB_vA, blkDim>>>(
    				
    	            (float *)d_CallResult,
    	            (float *)d_PutResult,
    	            (float *)d_StockPrice,
    	            (float *)d_OptionStrike,
    	            (float *)d_OptionYears,
    	            RISKFREE,
    	            VOLATILITY,
    	            OPT_N,
    	            TB_BS, blkDim, 
    	            d_A, d_B, d_C, numElements,
    	            TB_vA, blkDim, 
    	            iter_per_blockvA, iter_per_blockBS,
					num_blockvA_perSM, num_blockBS_perSM, 
					d_cont_SMs, d_SM_blocks_layout, 
					d_cont_BS, d_cont_vA
			);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			
			cudaEventElapsedTime(&time_par, start, stop);
			
			//printf("Time=%f\n", time_par);
			
			time_acc += time_par;
		}
			
		checkCudaErrors(cudaDeviceSynchronize());
		sdkStopTimer(&hTimer);
		*time = time_acc / (float)NUM_ITERATIONS;
	

		if (TB_BS>0) BS_end_kernel();
		if (TB_vA>0) VA_end_kernel();
			
		checkCudaErrors(cudaFree(d_cont_SMs));
		checkCudaErrors(cudaFree(d_SM_blocks_layout));
		checkCudaErrors(cudaFree(d_cont_BS));
		checkCudaErrors(cudaFree(d_cont_vA));

		
		return 0;
}

int  procesa_residente(int numSMs, int blkDim, int resident_blocks_per_SM, int BS_percentage)
{
	for (int i=92;i<124;i+=4){


		int iter_per_blockvA = 32*i;
		int iter_per_blockBS = 32*16;

		int num_blockvA_perSM = 4;//8;
		int num_blockBS_perSM = resident_blocks_per_SM - num_blockvA_perSM;//8;

		int TB_BS = num_blockBS_perSM * numSMs;
		int TB_vAdd = num_blockvA_perSM * numSMs;

		float time_SMK;
		Leaky_SMK(TB_BS, TB_vAdd, blkDim, &time_SMK, iter_per_blockvA, iter_per_blockBS,
					num_blockvA_perSM, num_blockBS_perSM, numSMs);



		float time_SMT;

		int th_simd = numSMs / 2; // Number of SMs executing BS blocks
		TB_BS = resident_blocks_per_SM * th_simd;
		TB_vAdd = numSMs * resident_blocks_per_SM - TB_BS;

		Leaky_SMT(TB_BS, TB_vAdd, blkDim, &time_SMT, iter_per_blockvA, iter_per_blockBS, th_simd, numSMs);
		
		/* Run BS and get execution time */

		float time_vA, time_BS;

		int iter_per_block = (iter_per_blockBS * TB_BS ) / (resident_blocks_per_SM * numSMs);
		TB_BS = resident_blocks_per_SM * numSMs;
		BS_start_kernel(TB_BS, blkDim, iter_per_block);
		printf("BS_solo --  TB_BS=%d, BlkDim=%d, I_BS=%d, size_BS=%d\n", TB_BS, blkDim, iter_per_block, OPT_N);
		BS_execute_kernel (TB_BS, blkDim, iter_per_block, &time_BS);
		BS_end_kernel();

		/** Run vA and get execution time */
		iter_per_block = (iter_per_blockvA * TB_vAdd ) / (resident_blocks_per_SM * numSMs);
		TB_vAdd = resident_blocks_per_SM * numSMs;
		VA_start_kernel(TB_vAdd, blkDim, iter_per_block); // Any number of TB can be used
		printf("vA_solo --  TB_BS=%d, BlkSim=%d, I_BS=%d, size_vAdd=%d\n", TB_vAdd, blkDim, iter_per_block, numElements);
		VA_execute_kernel (TB_vAdd, blkDim, iter_per_block, &time_vA);
		VA_end_kernel();

		// Execute only vA

		/*iter_per_blockvA = (32*num_blockvA_perSM/resident_blocks_per_SM)*i;//16;
		num_blockvA_perSM = resident_blocks_per_SM;

		TB_BS = num_blockBS_perSM * numSMs;
		TB_vAdd = num_blockvA_perSM * numSMs;

		Leaky_SMK(0, TB_vAdd, blkDim, &time_vA, iter_per_blockvA, 0,
					num_blockvA_perSM, 0, numSMs);
		*/

		// Execute only vA

		/*iter_per_blockBS = 16*32*num_blockBS_perSM/resident_blocks_per_SM;//16;
		num_blockBS_perSM = resident_blocks_per_SM;

		TB_BS = num_blockBS_perSM * numSMs;
		TB_vAdd = num_blockvA_perSM * numSMs;

		Leaky_SMK(TB_BS, 0, blkDim, &time_BS, 0, iter_per_blockBS,
					0, num_blockBS_perSM, numSMs);
		*/

		printf(" i=%d Speedup_SMK=%f Speedup_SMT=%f\n", i, (time_BS + time_vA)/time_SMK,  (time_BS + time_vA)/time_SMT);

	}

	return 0;

}


int procesa_no_residente(int numSMs, int blkDim, int max_block_per_SM, int BS_percentage)
{

	float time_vA, time_BS;
	
	int iter_per_blockBS=4;
	int TB_BS = numSMs * max_block_per_SM * 10;
	int TB_vAdd = (TB_BS * (100-BS_percentage))/BS_percentage;
	
	BS_start_kernel(TB_BS, blkDim, iter_per_blockBS);
	printf("BS_solo --  TB_BS=%d, BlkDim=%d, I_BS=%d, size_BS=%d\n", TB_BS, blkDim, iter_per_blockBS, OPT_N);
	BS_execute_kernel (TB_BS, blkDim, iter_per_blockBS, &time_BS);
	BS_end_kernel();
	

	for (int i=1;i<40;i+=4){
	
		int iter_per_blockvA = i;
		
		// SMK
		
		int num_blockBS_perSM = (max_block_per_SM  * BS_percentage)/100;
		int num_blockvA_perSM = max_block_per_SM - num_blockBS_perSM ;
		
		
		float time_SMK;
		Leaky_SMK(TB_BS, TB_vAdd, blkDim, &time_SMK, iter_per_blockvA, iter_per_blockBS,
				num_blockvA_perSM, num_blockBS_perSM, numSMs);
				
		
		// SMT
		
		int th_simd = (numSMs * BS_percentage) / 100;
		
		float time_SMT;
		Leaky_SMT(TB_BS, TB_vAdd, blkDim, &time_SMT, iter_per_blockvA, iter_per_blockBS, th_simd, numSMs);

		//float time_SMT0;
		//Leaky_SMT(TB_BS, 0, blkDim, &time_SMT0, iter_per_blockvA, iter_per_blockBS, numSMs, numSMs);

		
		/** Run vA and get execution time */
	
		VA_start_kernel(TB_vAdd, blkDim, iter_per_blockvA); // Any number of TB can be used
		printf("vA_solo --  TB_BS=%d, BlkDim=%d, iter_per_blockvA=%d, size_vAdd=%d\n", TB_vAdd, blkDim, iter_per_blockvA, numElements);
		VA_execute_kernel (TB_vAdd, blkDim, iter_per_blockvA, &time_vA);
		VA_end_kernel();
		
		printf(" i=%d Speedup_SMK=%f Speedup_SMT=%f \t SMK/SMT_Speedup=%f\n\n", i, (time_BS + time_vA)/time_SMK,  (time_BS + time_vA)/time_SMT, time_SMK/time_SMT);

	}
	
	return 0;
}

//////////////////////////////////////////////////////////
// For a specific granularity of each kernel
// search for the best SMs kernel mapping
/////////////////////////////////////////////////////////

int NR_SM_variation_no_residente(int numSMs, int blkDim, int max_block_per_SM)
{
	float time_vA, time_BS;

	/* Computational configuration per kernel **/
	
	int iter_per_blockBS=20; // Constant coarsening for BS
	int iter_per_blockvA=20;
	int TB_BS = max_block_per_SM*numSMs*10;
	int TB_vAdd = TB_BS*2;

	
	/** No CKE execution **/
	
	BS_start_kernel(TB_BS, blkDim, iter_per_blockBS);
	//intf("BS_solo --  TB_BS=%d, BlkDim=%d, I_BS=%d, size_BS=%d\n", TB_BS, blkDim, iter_per_blockBS, OPT_N);
	BS_execute_kernel (TB_BS, blkDim, iter_per_blockBS, &time_BS);
	BS_end_kernel();
	
	VA_start_kernel(TB_vAdd, blkDim, iter_per_blockvA); // Any number of TB can be used
	//printf("vA_solo --  TB_VA=%d, BlkDim=%d, I_VA=%d, size_VA=%d\n", TB_vAdd, blkDim, iter_per_blockvA, numElements);
	VA_execute_kernel (TB_vAdd, blkDim, iter_per_blockvA, &time_vA);
	VA_end_kernel();
	
	printf("SoloBStime=%f SoloVAtime=%f\n", time_BS, time_vA);

	printf("*************** Results for SMT ****************\n"); 
	
	for (int num_BS_SMs=1; num_BS_SMs < numSMs; num_BS_SMs++) {
		
			//SMT

			int th_simd = num_BS_SMs;

			float time_SMT;
			Leaky_SMT(TB_BS, TB_vAdd, blkDim, &time_SMT, iter_per_blockvA, iter_per_blockBS, th_simd, numSMs);

			printf("NUM_BS_SMs=%d SMT_Time%f Speedup_SMT=%f\n\n", num_BS_SMs, time_SMT, (time_BS + time_vA)/time_SMT);
	}

	printf("*\n************** Results for SMK ****************\n"); 

	for (int num_blockBS_perSM=1; num_blockBS_perSM < max_block_per_SM; num_blockBS_perSM++) {
	
		// SMK
		
		int num_blockvA_perSM = max_block_per_SM - num_blockBS_perSM;
		
		float time_SMK;
		Leaky_SMK(TB_BS, TB_vAdd, blkDim , &time_SMK, iter_per_blockvA, iter_per_blockBS,
				num_blockvA_perSM, num_blockBS_perSM, numSMs);
				
		printf("num_blockBS_perSM=%d num_blockvA_perSM=%d SMK_Time%f Speedup_SMK=%f\n\n", num_blockBS_perSM, num_blockvA_perSM, time_SMK, (time_BS + time_vA)/time_SMK);
	}
		
	return 0;

}

int NR_SM_variation_no_residente_2(int numSMs, int blkDim, int resident_block_per_SM)
{
	float time_vA, time_BS;
	int iter_per_blockvA, iter_per_blockBS, TB_BS, TB_vAdd;
	
	printf("*\n************** Results for SMT ****************\n"); 

	/* Computational configuration per kernel **/
	
	iter_per_blockBS=5; // Constant coarsening for BS
	
	for (int iter=1; iter<7; iter++){
	
		iter_per_blockvA= iter * iter_per_blockBS;
	
		for (int num_BS_SMs=1 ; num_BS_SMs < numSMs; num_BS_SMs++) {
	
			TB_BS = num_BS_SMs*200;
			TB_vAdd = (resident_block_per_SM - num_BS_SMs)*200;
	
			// No CKE execution 
	
			BS_start_kernel(TB_BS, blkDim, iter_per_blockBS);
			//intf("BS_solo --  TB_BS=%d, BlkDim=%d, I_BS=%d, size_BS=%d\n", TB_BS, blkDim, iter_per_blockBS, OPT_N);
			BS_execute_kernel (TB_BS, blkDim, iter_per_blockBS, &time_BS);
			BS_end_kernel();
	
			VA_start_kernel(TB_vAdd, blkDim, iter_per_blockvA); // Any number of TB can be used
			//printf("vA_solo --  TB_VA=%d, BlkDim=%d, I_VA=%d, size_VA=%d\n", TB_vAdd, blkDim, iter_per_blockvA, numElements);
			VA_execute_kernel (TB_vAdd, blkDim, iter_per_blockvA, &time_vA);
			VA_end_kernel();
	
			//printf("SoloBStime=%f SoloVAtime=%f\n", time_BS, time_vA);

			//SMT

			int th_simd = num_BS_SMs;

			float time_SMT;
			Leaky_SMT(TB_BS, TB_vAdd, blkDim, &time_SMT, iter_per_blockvA, iter_per_blockBS, th_simd, numSMs);
			
			float Sup = (time_BS + time_vA)/time_SMT;

			if (Sup > 1.05)
				printf("iter =%d NUM_BS_SMs=%d NUM_vA_SMs=%d SMT_Time%f Speedup_SMT=%f\n\n", iter, num_BS_SMs, resident_block_per_SM - num_BS_SMs, time_SMT, Sup);
		}
	}

	printf("*\n************** Results for SMK ****************\n"); 
	
	/* Computational configuration per kernel **/
	
	iter_per_blockBS=5; // Constant coarsening for BS
	
	for (int iter=1; iter<7; iter++){
	
		iter_per_blockvA= iter * iter_per_blockBS;

		for (int num_blockBS_perSM=1; num_blockBS_perSM < resident_block_per_SM; num_blockBS_perSM++) {
	
			// SMK
		
			int num_blockvA_perSM = resident_block_per_SM - num_blockBS_perSM;
			
			TB_BS = num_blockBS_perSM*numSMs*50;
			TB_vAdd = num_blockvA_perSM*numSMs*50;
	
			/** No CKE execution **/
	
			BS_start_kernel(TB_BS, blkDim, iter_per_blockBS);
			//intf("BS_solo --  TB_BS=%d, BlkDim=%d, I_BS=%d, size_BS=%d\n", TB_BS, blkDim, iter_per_blockBS, OPT_N);
			BS_execute_kernel (TB_BS, blkDim, iter_per_blockBS, &time_BS);
			BS_end_kernel();
	
			VA_start_kernel(TB_vAdd, blkDim, iter_per_blockvA); // Any number of TB can be used
			//printf("vA_solo --  TB_VA=%d, BlkDim=%d, I_VA=%d, size_VA=%d\n", TB_vAdd, blkDim, iter_per_blockvA, numElements);
			VA_execute_kernel (TB_vAdd, blkDim, iter_per_blockvA, &time_vA);
			VA_end_kernel();
		
			float time_SMK;
			Leaky_SMK(TB_BS, TB_vAdd, blkDim , &time_SMK, iter_per_blockvA, iter_per_blockBS,
					num_blockvA_perSM, num_blockBS_perSM, numSMs);
				
			float Sup = (time_BS + time_vA)/time_SMK;

			if (Sup > 1.05)
				printf("iter =%d  num_blockBS_perSM=%d  num_blockvA_perSM=%d SMK_Time%f Speedup_SMK=%f\n\n", iter,  num_blockBS_perSM,  num_blockvA_perSM, time_SMK, Sup);
				
		}
		
	}
		
	return 0;

}





int NR_SM_variation_residente(int numSMs, int blkDim, int Blq_resi_perSM)
{
	float time_vA, time_BS;
	int TB_BS, TB_vAdd, iter_per_blockvA, iter_per_blockBS;

	int iter_basevA = 300;
	int iter_baseBS = 50;
	 
	/** No CKE execution **/
	
	TB_BS = numSMs * Blq_resi_perSM;
	iter_per_blockBS = iter_baseBS;
		
	BS_start_kernel(TB_BS, blkDim, iter_per_blockBS);
	printf("BS_solo --  TB_BS=%d, BlkDim=%d, I_BS=%d, size_BS=%d\n", TB_BS, blkDim, iter_per_blockBS, OPT_N);
	BS_execute_kernel (TB_BS, blkDim, iter_per_blockBS, &time_BS);
	BS_end_kernel();
		
	TB_vAdd = numSMs * Blq_resi_perSM;
	iter_per_blockvA = iter_basevA;
	
	VA_start_kernel(TB_vAdd, blkDim, iter_per_blockvA); // Any number of TB can be used
	printf("vA_solo --  TB_VA=%d, BlkDim=%d, I_VA=%d, size_VA=%d\n", TB_vAdd, blkDim, iter_per_blockvA, numElements);
	VA_execute_kernel (TB_vAdd, blkDim, iter_per_blockvA, &time_vA);
	VA_end_kernel();	
	
	printf("SoloBStime=%f SoloVAtime=%f\n", time_BS, time_vA);
	
	printf("*************** Results for SMT ****************\n"); 
	
	printf("TB_BS \t TB_vAdd \t Speedup\n");
	for (int num_BS_SMs=1; num_BS_SMs < numSMs; num_BS_SMs++) {
		
			//SMT

		int th_simd = num_BS_SMs;

		TB_BS = num_BS_SMs * Blq_resi_perSM;
		iter_per_blockBS = (int) round((float)(numSMs * iter_baseBS)/(float)num_BS_SMs);
	
		TB_vAdd = (numSMs - num_BS_SMs) * Blq_resi_perSM;
		iter_per_blockvA = (int)round((float)(numSMs * iter_basevA)/(float)(numSMs - num_BS_SMs));
		
		float time_SMT;
		Leaky_SMT(TB_BS, TB_vAdd, blkDim, &time_SMT, iter_per_blockvA, iter_per_blockBS, th_simd, numSMs);
		
		printf("%d \t %d \t \t %.2f %f %f\n", TB_BS, TB_vAdd, (time_BS + time_vA)/time_SMT, time_BS, time_vA);
	}

	printf("*\n************** Results for SMK ****************\n"); 
	printf("TB_BS \t TB_vAdd \t Speedup\n");

	for (int num_blockBS_perSM=1; num_blockBS_perSM <Blq_resi_perSM; num_blockBS_perSM++) {
	
		// SMK
		
		int num_blockvA_perSM = Blq_resi_perSM - num_blockBS_perSM;
		
		TB_BS = num_blockBS_perSM * numSMs;
		iter_per_blockBS = (int) round((float)(Blq_resi_perSM * iter_baseBS)/(float)num_blockBS_perSM);
		
		TB_vAdd = num_blockvA_perSM * numSMs;
		iter_per_blockvA = (int)round((float)(Blq_resi_perSM * iter_basevA)/(float)(num_blockvA_perSM));
		
		float time_SMK;
		Leaky_SMK(TB_BS, TB_vAdd, blkDim , &time_SMK, iter_per_blockvA, iter_per_blockBS,
				num_blockvA_perSM, num_blockBS_perSM, numSMs);
				
		printf("%d \t %d \t \t %.2f %f %f \n", TB_BS, TB_vAdd, (time_BS + time_vA)/time_SMK, time_BS, time_vA);
	}
		
	return 0;

}



int  calcula_overhead(int numSMs, int blkDim, int resident_blocks_per_SM, int BS_percentage)
{

		
		/** Computational configuration per kernel **/
		
		int iter_per_blockBS=10;
		int iter_per_blockvA=10;

		int TB_BS = resident_blocks_per_SM*numSMs*10;
		int TB_vAdd = resident_blocks_per_SM*numSMs*10;
 
		BS_start_kernel(TB_BS, blkDim, iter_per_blockBS);
		printf("BS_solo --  TB_BS=%d, BlkDim=%d, I_BS=%d, size_BS=%d\n", TB_BS, blkDim, iter_per_blockBS, OPT_N);
		float time_BS;
		BS_execute_kernel (TB_BS, blkDim, iter_per_blockBS, &time_BS);
		BS_end_kernel();
		//printf("BS=%f\n", time_BS);

		VA_start_kernel(TB_vAdd, blkDim, iter_per_blockvA); // Any number of TB can be used
		printf("vA_solo --  TB_BS=%d, BlkDim=%d, iter_per_blockvA=%d, size_vAdd=%d\n", TB_vAdd, blkDim, iter_per_blockvA, numElements);
		float time_vA;
		VA_execute_kernel (TB_vAdd, blkDim, iter_per_blockvA, &time_vA);
		VA_end_kernel();
		//printf("vA=%f\n", time_vA);

	/*	iter_per_blockBS=12;
		iter_per_blockvA=48;

		TB_BS = resident_blocks_per_SM*numSMs*10;
			TB_vAdd = resident_blocks_per_SM*numSMs*10;

				BS_start_kernel(TB_BS, blkDim, iter_per_blockBS);
				//printf("BS_solo --  TB_BS=%d, BlkDim=%d, I_BS=%d, size_BS=%d\n", TB_BS, blkDim, iter_per_blockBS, OPT_N);
				BS_execute_kernel (TB_BS, blkDim, iter_per_blockBS, &time_BS);
				BS_end_kernel();
				printf("BS=%f\n", time_BS);

				VA_start_kernel(TB_vAdd, blkDim, iter_per_blockvA); // Any number of TB can be used
				//printf("vA_solo --  TB_BS=%d, BlkDim=%d, iter_per_blockvA=%d, size_vAdd=%d\n", TB_vAdd, blkDim, iter_per_blockvA, numElements);

				VA_execute_kernel (TB_vAdd, blkDim, iter_per_blockvA, &time_vA);
				VA_end_kernel();
				printf("vA=%f\n", time_vA);
*/
		// SMK

		float time_SMK_BS;
		Leaky_SMK(TB_BS, 0, blkDim, &time_SMK_BS, 0, iter_per_blockBS,0, resident_blocks_per_SM, numSMs);

		float time_SMK_vA;
		Leaky_SMK(0, TB_vAdd, blkDim, &time_SMK_vA, iter_per_blockvA, 0, resident_blocks_per_SM, 0, numSMs);


		float time_SMT_BS;
		Leaky_SMT(TB_BS, 0, blkDim, &time_SMT_BS, 0, iter_per_blockBS, resident_blocks_per_SM, numSMs);

		float time_SMT_vA;
		Leaky_SMT(0, TB_vAdd, blkDim, &time_SMT_vA, iter_per_blockvA, 0, 0, numSMs);

		printf("Time BS(ms) Solo_BS=%f SMK_BS=%f SMT_BS=%f \n", time_BS, time_SMK_BS, time_SMT_BS);
		printf("Time vA(ms) Solo_vA=%f SMK_vA=%f SMT_vA=%f \n", time_vA, time_SMK_vA, time_SMT_vA);

		printf(" Overhead: SMK_BS=%f SMK_vA=%f SMT_BS=%f SMT_vA=%f\n",(time_SMK_BS-time_BS)/time_BS,  (time_SMK_vA-time_vA)/time_vA, (time_SMT_BS-time_BS)/time_BS,  (time_SMT_vA-time_vA)/time_vA);




		// SMK
/*
		int num_blockBS_perSM = (resident_blocks_per_SM * BS_percentage)/100;//8;
		int num_blockvA_perSM = resident_blocks_per_SM - num_blockBS_perSM;//8;

		int TB_BS = \n"num_blockBS_perSM * numSMs;
		int TB_vAdd = num_blockvA_perSM * numSMs;

		float time_SMK;
		Leaky_SMK(TB_BS, TB_vAdd, blkDim, &time_SMK, iter_per_blockvA, iter_per_blockBS,
							num_blockvA_perSM, num_blockBS_perSM, numSMs);
*/
		//SMT
/*
		int th_simd = (numSMs * BS_percentage) / 100; // Number of SMs executing BS blocks
		TB_BS = resident_blocks_per_SM * th_simd;
		TB_vAdd = numSMs * resident_blocks_per_SM - TB_BS;

		float time_SMT;
		Leaky_SMT(TB_BS, TB_vAdd, blkDim, &time_SMT, iter_per_blockvA, iter_per_blockBS, th_simd, numSMs);

		printf(" i=%d Speedup_SMK=%f Speedup_SMT=%f\n", i, (time_BS + time_vA)/time_SMK,  (time_BS + time_vA)/time_SMT);
*/
		return 0;

}

///////////////////////////////////////////////////////
//Executing Leavy bucket for BS and VA
///////////////////////////////////////////////////////

int main()
{ 
	// Computation of BS id fixed. Computation of VA depends on the SM distribution

	// Establish main parameters
	
// Select device
	
	
	int dev = 0, numSMs, blkDim, blkDim_BS, blkDim_vA, resident_blocks_per_SM;
	
	cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	
	if (dev == 0) {
		numSMs= 13; // K20c
		blkDim = 128;
		blkDim_BS = blkDim;
		blkDim_vA= blkDim;
		resident_blocks_per_SM = 16; // Depends on the code: check occupancy calculator
	}
	else {
		numSMs= 16; // GTX580
		blkDim = 128;
		blkDim_BS = blkDim;
		blkDim_vA= blkDim;
		resident_blocks_per_SM = 8; // Depends on the code: check occupancy calculator
	}
	
	
	//calcula_overhead(numSMs, blkDim, resident_blocks_per_SM, 50);
 
 	//NR_SM_variation_residente(numSMs, blkDim, resident_blocks_per_SM);

	NR_SM_variation_no_residente_2(numSMs, blkDim, resident_blocks_per_SM);

	//procesa_residente(numSMs, blkDim, resident_blocks_per_SM, 50);

	//procesa_no_residente(numSMs, blkDim, resident_blocks_per_SM, 50
	
	


	return 0;
		
}
		
		

		
