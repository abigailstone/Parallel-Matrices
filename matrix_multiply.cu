/*
 * CUDA matrix multiply
 */

#include <vector>
#include <iostream>
#include <sstream>
#include <cassert>
#include <chrono>

#include "utils.hxx"
#include "bits.hxx"
#include "matrix.hxx"

using namespace std;

const int BS = 32;	// The blocks are BS x BS.

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// Simple algorithm for multiplying two matrices.
void Matrix::matmul_naive(const Matrix &A, const Matrix &B) {
    int N = this->N();
    for (int r=0; r<N; ++r){
	    for (int k=0; k<N; ++k){
	        for (int c=0; c<N; ++c) {
		        if (k==0){
		            (*this)(r,c) = 0.0F;
                }
		        (*this)(r,c) += (A(r,k) * B(k,c));
	        }
        }
    }
}

// CUDA kernel function 
__global__ void matmul_kernel (float *d_A, float *d_B, float *d_C, int N) {

    int rB = blockIdx.x; 
    int cB = blockIdx.y;  

    int ri = threadIdx.x; 
    int ci = threadIdx.y; 

    __shared__ float SA[BS][BS]; 
    __shared__ float SB[BS][BS];

    // printf("In thread with r=(%d,%d) c=(%d,%d)\n", rB,ri,cB,ci);   

    // get row and column locations 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    int sum = 0;

    for (int kB = 0; kB < gridDim.x; kB++){  
        
        // coalesced memory access
        SA[ci][ri] = d_A[row*N + (kB*BS + ri)];
        SB[ci][ri] = d_B[(kB*BS + ci)*N + col];

        __syncthreads(); 

        for (int ki = 0; ki < BS; ki++){ 
            // shared-memory bank access 
            sum += SA[ci][ki] * SB[ki][ri];
        }  

        __syncthreads();

    }   

    // store results back into device memory
    d_C[row * N + col] = sum;

}

// host function
// allocates memory and moves data between CPU and GPU
void Matrix::matmul_host (const Matrix &A, const Matrix &B, int BS) {

    int numElem=N()*N();
    int sizeBytes = numElem*4;

    // Copy A from host memory to device memory.
    float *d_A = NULL;
    cudaError_t err = cudaMalloc((void **)&d_A, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix A");

    err = cudaMemcpy (d_A, A.data.data(), sizeBytes, cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix A from host to device");

    // Allocate device memory for B.
    float *d_B = NULL; 
    err = cudaMalloc((void **)&d_B, sizeBytes); 
    ERR_CHK (err, "Failed to allocate device matrix B");

    // Copy B from host memory to device memory.
    err = cudaMemcpy(d_B, B.data.data(), sizeBytes, cudaMemcpyHostToDevice); 
    ERR_CHK(err, "Failed to copy matrix B from host to device");

    // Allocate device memory for C. 
    float *d_C = NULL; 
    err = cudaMalloc((void **)&d_C, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix C");

    // Launch the CUDA Kernel
    dim3 thBlocks(N()/BS, N()/BS); 
    dim3 threads(BS, BS); 

    // matrix multiplication
    matmul_kernel <<< thBlocks, threads >>> (d_A, d_B, d_C, N());

    // Copy the result from device memory to host memory. 
    err = cudaMemcpy(this->data.data(), d_C, sizeBytes, cudaMemcpyDeviceToHost);
    ERR_CHK (err, "Failed to copy result C from device to host");

    // Free device memory.
    err = cudaFree(d_A);
    ERR_CHK (err, "Failed to free CUDA matrix A");
    
    err = cudaFree(d_B);
    ERR_CHK (err, "Failed to free CUDA matrix B");

    err = cudaFree(d_C);
    ERR_CHK (err, "Failed to free CUDA matrix C");
}


// This function executes the various pieces for a given matrix size.
static void run (int N) {

    Matrix a(N), b(N), c(N), d(N);

    // initialize A and B 
    a.init_random(5);
    b.init_random(10);
    //a.init_cyclic_order();
    //b.init_count_order();
    //b.init_identity();

    LOG (endl<<"Working on "<<N<<"x"<<N<<" matrices with BS="<<BS);
    //LOG ("A="<<a.str());
    //LOG ("B="<<b.str());

    // Compute the reference solution
    auto start = start_time();
    c.matmul_naive(b, a);
    long int time = delta_usec (start);
    LOG ("Naive matrix multiplication took "<<(time/1000000.0)<<"sec");
 
    long int total_time=0;

    for (int i=0; i<4; ++i) {
	    auto start = start_time();

	    d.matmul_host (b, a, BS);

	    long int time = delta_usec (start);
	    total_time += time;
	    c.compare (d);
    }

    LOG ("matmul_host averaged "<<(total_time/4000000.0)<<"sec");
}

// Main() lives on the CPU.
int main() {

    // Check the compute capability
    cudaDeviceProp prop; 
    int device=0;

    cudaError_t err = cudaGetDeviceProperties (&prop, device);
    ERR_CHK (err, "Failed to get compute capability");
    int major = prop.major, minor=prop.minor;
    LOG ("Compute capability = "<<major<<"."<<minor);

    // Test a bunch of different matrix sizes 
    run (32);	
    run (64);	
    run (256);
    run (1024);	
    run (2048);	
    run (4096);
    	
    return (0);
}
