/* Parallelization: Histogram CUDA
 * By Maria Pantoja
 * June 2020
 * Parallel code -- CUDA Implementation of Histogram of characters
 *Input: Text file
 *Output: Histogram for the 256 ASCII characters on the file
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
     
    
/*N is the size of the file, number of characters in the file*/
#define N    (100*1024*1024) 

/*use this function to generate a random character file of size 
  students can read a bif file from wikipedia or any other source*/
void* random_block( int size ) {
    unsigned char *data = (unsigned char*)malloc( size );  
    if(data == NULL) {
	printf("error allocating big file");
	exit(1);
    }
    for (int i=0; i<size; i++)
        data[i] = rand();
    return data;
}

__global__ void histoNoShared( unsigned char *buffer,long size,unsigned int *histo ) {
    //global thread index   
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    //each thread performs just one operation reads buffer and increment the specific bucket
    //needs atomicadd since more than one thread can write to same bucket
    if (i < size) {
        atomicAdd( &histo[buffer[i]], 1 );
    }
}

__global__ void histoNoSharedStride( unsigned char *buffer,long size,unsigned int *histo ) {
    // calculate global index
    // calculate stride as the total number of threads running (#blocks*#threadsperblock)
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    //each thread performs size/(numberblocks*numberthreadsperblock) operations, 
    //reads buffer and increment the specific bucket
    //needs atomicadd since more than one thread can write to same bucket
    while (i < size) {
        atomicAdd( &histo[buffer[i]], 1 );
        i += stride;
    }
}
__global__ void HistoSharedMem( unsigned char *buffer, long size,unsigned int *histo )
{
    int iGlobal = threadIdx.x + blockIdx.x * blockDim.x;
    int iLocal= threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    __shared__ unsigned int temp[256];
    temp[iLocal] = 0;
    __syncthreads();

    //atomic operation per block in shared memory
    while (iGlobal < size){
        atomicAdd( &temp[buffer[iGlobal]], 1);
        iGlobal += offset;
    }
    __syncthreads();
    //atomic operation in global memory
    atomicAdd( &(histo[iLocal]), temp[iLocal] );
}

void HistoCPU( unsigned char *buffer,long size, unsigned int *hist ) {
    //initialize the buckets for the histogram to 0
    for (int i=0; i<256; i++)
        hist[i] = 0;
    //calculate the histogram on the CPU
    for (int i=0; i<size; i++)
        hist[buffer[i]]++;
}

int main( void ) {
    clock_t start, end;
    double cpu_time_used;
    
    unsigned char *buffer = (unsigned char*)random_block( N );
    //create Histogram on CPU
    unsigned int histCPU[256];
    start = clock();
    HistoCPU(buffer,N, histCPU );
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    //print time in seconds on CPU
    printf("HistCPU took %f seconds to execute \n", cpu_time_used); 
    // create HIstogram on the GPU
    // allocate memory on the GPU for the file's data
    unsigned char *dev_buffer;
    unsigned int *dev_hist;
    cudaMalloc( (void**)&dev_buffer, N );
    cudaMemcpy( dev_buffer, buffer, N,cudaMemcpyHostToDevice );
    cudaMalloc( (void**)&dev_hist,256 * sizeof( int ) ) ;
    cudaMemset( dev_hist, 0,256 * sizeof( int ) ) ;

    int deviceId;
    // `deviceId` now points to the id of the currently active GPU.
    cudaGetDevice(&deviceId);
    // `props` now has many useful properties about the active GPU device.
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    //number of bloks should be the 2*number of streaming multiprocessors
    int blocks = props.multiProcessorCount;
    //kernel call
    //histoNoShared<<<N/256,256>>>( dev_buffer, N, dev_hist );
    //histoNoSharedStride<<<blocks*2,256>>>( dev_buffer, N, dev_hist );
    HistoSharedMem<<<blocks*2,256>>>( dev_buffer, N, dev_hist );
    //copy back results to the CPU
    unsigned int    histGPU[256];
    cudaMemcpy( histGPU, dev_hist,256 * sizeof( int ),cudaMemcpyDeviceToHost ) ;

    // verify that we have the same counts via CPU
    for (int i=0; i<256; i++) {
        if (histGPU[i] != histCPU [i]){
		printf( "Failure at %d!  Off by %d\n", i, histGPU[i] );
	}
    }
    printf("DONE, SUCESS!!!\n");
    cudaFree( dev_hist );
    cudaFree( dev_buffer );
    free( buffer );
    return 0;
}
