/*
 * CUDA convolution
 */

#include <iostream>
#include <assert.h>
#include <iomanip> 

#include "utils.hxx"
#include "matrix.hxx"

#include <cuda_runtime.h> 

using namespace std;

const int BS = 32;		// block size
const int FILT_SIZE_MAX = 12;	// filter size

// Simple convolution algorithm. 1 core, no blocking. Used as a reference to check
// the GPU version
// Inputs: array_in (the input), f (the filter)
// Outputs: set this' data (which should already be allocated to the correct
//	    size, slightly smaller than array_in.
// Assume: 1 input channel, 1 output channel.
void Matrix::conv_naive (const Matrix &array_in, const Matrix &f) {

    assert (this->N() == array_in.N() + 1 - f.N());
    int N = this->N();
    
    // Iterate over output rows and columns
    for (int ro=0; ro<N; ++ro){		
        for (int co=0; co<N; ++co) {	
            
            float sum=0;
            
            // iterate over filter rows and columns
            for (int rf=0; rf<f.N(); ++rf){		
                for (int cf=0; cf<f.N(); ++cf) {
                    int ri = ro + rf;
                    int ci = co + cf;
                    sum += array_in(ri,ci) * f(rf,cf);
                }
            }

            (*this)(ro, co) = sum; 

        } 
    }
}


// CUDA kernel function
__global__ void conv_cuda_kernel (float *d_inp, float *d_f, float *d_out, int Nin, int Nf, int Nout) { 
    
    int rB = blockIdx.x; 
    int cB = blockIdx.y; 
    
    int ri = threadIdx.x; 
    int ci = threadIdx.y;

    // get row and coolumn locations 
    int row = blockIdx.y * (BS - Nf + 1) + threadIdx.y; 
    int col = blockIdx.x * (BS - Nf + 1) + threadIdx.x;  

    int rowi = row - Nf + 1; 
    int coli = col - Nf + 1; 

    __shared__ float Sin[BS][BS];  

    // load block from device memory 
    if (rowi < Nin && rowi >= 0 && coli < Nin && coli >= 0){
        Sin[threadIdx.y][threadIdx.x] = d_inp[coli * Nin + rowi];
    } else { 
        Sin[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    float sum = 0; 

    int maxTh  = BS - Nf + 1;
    int maxOut = Nout - Nf + 1;

    if (threadIdx.y < maxTh && threadIdx.x < maxTh && row < maxOut && col < maxOut){
        for (int fr = 0; fr < Nf; fr++){
            for (int fc = 0; fc  < Nf; fc++){
                sum += Sin[threadIdx.y + fr][threadIdx.x + fc] * d_f[fc*Nf + fr];
            }
        } 

        __syncthreads();

        d_out[col * Nout + row] = sum;
    }

}

// host function - allocates memory and moves data between CPU and GPU
void Matrix::conv_host (const Matrix &inp, const Matrix &f, int dummy) {

    auto start1 = start_time();

    // Allocate input matrix in device memory
    assert (1<<inp._log2NColsAlc == inp._nCols);

    int numElem = inp.data.size();
    int sizeBytes = numElem*4;
    float *d_inp = NULL;
    cudaError_t err = cudaMalloc((void **)&d_inp, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix 'inp'");

    // Copy input from host memory to device memory.
    err = cudaMemcpy (d_inp, inp.data.data(), sizeBytes,cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix inp from host to device");

    // Allocate device memory for filter
    float *d_f = NULL;
    sizeBytes = static_cast<int> (f.data.size()) * 4;
    err = cudaMalloc((void **)&d_f, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix for the filter f");

    // Copy f from host memory to device memory.
    err = cudaMemcpy (d_f, f.data.data(), sizeBytes, cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix f from host to device");

    // Allocate device memory for the output matrix.
    int width = inp._nRows*4; // in bytes 
    int height = inp._nRows; // in rows  
    size_t pitch;

    float *d_out = NULL; 
    sizeBytes = (inp._nRows - f._nRows + 1);

    err = cudaMalloc((void**)&d_out, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix 'out'");
    cudaDeviceSynchronize(); long int time1 = delta_usec (start1);
    auto start2 = start_time();

    //////////////////////////////////
    // Launch the CUDA Kernel 
    int Nin = inp._nRows; 
    int Nf = f._nRows; 
    int Nout = (Nin - Nf + 1);

    int nBlocks = inp._nRows / BS;   
    dim3 grid(nBlocks, nBlocks); // create thread blocks in one grid 
    dim3 block(BS, BS); // create threads in each thread block 

    conv_cuda_kernel <<<grid, block>>> (d_inp, d_f, d_out, Nin, Nf, Nout);  
    err = cudaGetLastError();
    ERR_CHK (err, "Failed to launch or finish CNN_kernel"); 

    ///////////////////////////////////

    cudaDeviceSynchronize(); long int time2 = delta_usec (start2);
    auto start3 = start_time();

    // Copy the result from device memory to host memory.
    err = cudaMemcpy(this->data.data(), d_out, sizeBytes,cudaMemcpyDeviceToHost);
    ERR_CHK (err, "Failed to copy result from device to host");
    cudaDeviceSynchronize(); long int time3 = delta_usec (start3);

    err = cudaFree(d_inp);
    ERR_CHK (err, "Failed to free CUDA matrix inp");
    err = cudaFree(d_f);
    ERR_CHK (err, "Failed to free CUDA matrix f");
    err = cudaFree(d_out);
    ERR_CHK (err, "Failed to free CUDA matrix out");

    cout << setprecision(3) << fixed;
    LOG ("\tCUDA " <<inp.nRows()<<"x"<<inp.nRows()
	 << " CNN with "<<f.nRows()<<"x"<<f.nRows()<<" filter took "
	 <<(time1+time2+time3)/1000000.0<<" sec; "<<(time1/1000000.0)<<"s copy to, "
	 << (time2/1000000.0)<<"s for computation, "<< (time3/1000000.0)<<"s copy back ");
}



static void run (int mat_size, int filt_size) {
    LOG ("Running "<<mat_size<<"x"<<mat_size<<" with "
		<<filt_size<<"x"<<filt_size<<" filter");

    // Create the filter matrix.
    Matrix f(filt_size);
    //f.init_identity();
    f.init_random (5);

    // Create input and output matrices.
    Matrix in(mat_size);
    //in.init_cyclic_order();
    //in.init_identity();
    in.init_random(5);
    Matrix out(mat_size+1-filt_size);

    // Run and time the single-thread CPU algorithm.
    auto start = start_time();
    Matrix naive(mat_size+1-filt_size);
    naive.conv_naive (in, f);
    long int time = delta_usec (start);
    LOG ("\t"<<mat_size<<"x"<<mat_size<<" conv_naive() took "<<(time/1000000.0)<<"sec");

    // Run and time the GPU version.
    time=0;

    for (int rep=0; rep<3; ++rep) {
	    out.init_identity(); // re-initalize 

	    auto start = start_time();
	    out.conv_host (in, f, 0);
        
	    long int dt = delta_usec (start);
	    time += dt;
	    naive.compare (out);
    }
    LOG ("    "<<mat_size<<"x"<<mat_size<<" CUDA CNN took "<<(time/3000000.0)
		<<" sec on average.");
}

int main () {

    // Check the compute capability
    cudaDeviceProp prop; int device=0;
    cudaError_t err = cudaGetDeviceProperties (&prop, device);
    ERR_CHK (err, "Failed to get compute capability");
    
    int major = prop.major, minor=prop.minor;
    LOG ("Compute capability = "<<major<<"."<<minor);

    // Run convolution with a bunch of matrix sizes and filter sizes
    run (32, 2);
    run (64, 2);
    run (4096, 4);
    run (4096, 8);
    run (4096, 12);
    run (8192, 4);
    run (8192, 8);
    run (8192, 12);
}
