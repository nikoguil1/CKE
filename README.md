This project develop a macrokernel to test the performance achievedby concurrent kernel execution. It implements to ways of distributing blocks from two different kernels:
- SMT: Spatial distribution -> blocks of different kernels are execued, if possible, in different SMs
- SMK: Blocks of different kernels are executed in the same SM using fix distribution, if possible, of blocks of each kernel. All the SMs will execute blocks using a similar distribution.  
