#define BS_COARSENING 1//16
#define VA_COARSENING 1 //32 

int BS_start_kernel(int TB_Number, int Blk_Size, int iter_per_block);
int VA_start_kernel(int TB_Number, int Blk_Size, int iter_per_block);

int BS_execute_kernel (int TB_Number, int Blk_Size, int iter_per_block, float *time);
int VA_execute_kernel(int TBN, int Blk_Size, int iter_per_block, float *time);

