# CUDA Matrix Fun


Parallel computing experiments for mutliplying matrices (`matrix_multiply.cu`) and simple 2D convolution (`convolution.cu`) using CUDA. 


Compile matrix multiplication: 
```
nvcc -O2 -std=c++11 matrix_multiply.cu matrix.cxx utils.cxx
``` 

Compile convolution: 
```
nvcc -O2 -std=c++11 convolution.cu matrix.cxx utils.cxx
```

To run on the GPU cluster: 

```
module load cuda/10.0
module load gcc/7.3.0
./a.out > results.out
```


## GPU Matrix Multiplication  

| Matrix Size | Naïve Matrix Multiplication | Parallel Matrix Multiplication | 
| --- | --- | --- | 
| 32 x 32 | 0.000047 s| 0.06530 s | 
| 64 x 64 | 0.00036 s | 0.00038 s| 
| 256 x 256 | 0.0216 s | 0.00073 s | 
|1024 x 1024 | 1.370 s| 0.01121 s| 
| 2048 x 2048 | 11.055 s| 0.06429 s | 
| 4096 x 4096 | 89.577 s | 0.25593 s |
 

 ## GPU Convolution 

 To address edge effects, one thread per input tile is used, and only certain threads write to the final output. Each streaming multiprocessor loads a 32 x 32 input block (for block size 32), and then computes a 32 x 32 output array in which only certain values are written to the final output. 

 For a square input matrix with $N_i$ rows and a square filter with $N_f$ rows, the number of rows in the output matrix is $N_o = N_i - N_f$.  

 For each output block of size $B_o$, we need to move $(B_o + N_f)^2$ input elements into shared memory. Each output entry needs to access $N_f^2$ input elements for a filter of size $N_f$. Using the tiled approach to memory access decreases the number of global memory access by converting them to shared memory accesses.