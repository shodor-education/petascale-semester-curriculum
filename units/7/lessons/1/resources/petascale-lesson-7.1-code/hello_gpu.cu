/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 7: CUDA
 * Lesson 1: Introduction to CUDA GPGPU
 * File: hello_gpu.c
 * Developed by Michael D. Shah for the Shodor Education Foundation, Inc.
 *
 * Copyright (c) 2020 The Shodor Education Foundation, Inc.
 *
 * Browse and search the full curriculum at
 * <http://shodor.org/petascale/materials/semester-curriculum>.
 *
 * We welcome your improvements! You can submit your proposed changes to this
 * material and the rest of the curriculum in our GitHub repository at
 * <https://github.com/shodor-education/petascale-semester-curriculum>.
 *
 * We want to hear from you! Please let us know your experiences using this
 * material by sending email to petascale@shodor.org
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// INSTRUCTIONS HOW TO RUN
//
// Compile with: nvcc hello_gpu.cu -o hello_gpu
// Run with    : ./hello_gpu
//
// Include our C Standard Libraries
#include <stdio.h>
#include <stdlib.h>

// This is our problem size
#define SIZE (640*480)

// Note that CUDA functions cannot return a value,
// so we declare them as void
//
// __global__ tells us that the function
// is callable from both the host and device.
__global__ void gpuAddOne(int* array, int size){
    // We need some  unique variable identifying which thread accesses
    // a piece of data. So below we are going to figure out which
    // thread (i.e. index) is going to operate on which piece of data.
    //
    // This gives us full coverage of every id that we want to access.
    // Figure out which block we are, what dimension, and which thread.
    // You can think of these values like a phone number if you like.
    // The block is the area code for example.
    int currentThread = (blockIdx.x * blockDim.x) + threadIdx.x;   
    // Do a bounds check to make sure we are not accessing
    // out of bounds memory.
    // (This can occur if we don't have a perfect number within 32.
    if(currentThread < size){
        array[currentThread] += 1;
    }
}



int main(){
    // ====================== CPU Code ====================
    // Create an array of numbers
    int* myCPUArray = (int*)malloc(SIZE*sizeof(int));
    // Let's initialize our values to a specific value to start.
    // We can do so using an array.
    for(int i=0; i < SIZE; i++){
        myCPUArray[i] = 0;
    }
    // ====================== CPU Code ====================



    
    // ====================== GPU Code ====================
    
    // Now let's allocate some memory for another array.
    // This memory will store the result of operations that occur
    // on the GPU
    int* myGPUArray = (int*)malloc(SIZE*sizeof(int));

    // Allocate memory for our device
    cudaMalloc(&myGPUArray, SIZE*sizeof(int));

    // We need to make our memory accessible to the GPU.
    // Now we are going to copy data from our CPU onto our GPU.
    cudaMemcpy(myGPUArray, myCPUArray, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    // Threadblock size
    // Typicaly we choose something that is a multiple of 32.
    int NUM_THREADS = 256;
    // Grid size
    // Note that our problem size is 640*480
    // So that means we need at least that many blocks, which 
    // each have 256 (i.e. NUM_THREADS) in order to solve
    // our problem.
    // Note: The ceil function rounds up from a floating point
    // number, hence why we divide by a float, to ensure we
    // have enough blocks to solve our problem.
    int NUM_BLOCKS = (int)ceil(SIZE/ (float)NUM_THREADS);

    // Now we are ready to execute our kernel
    // Between the <<< and >>> are our configuration parameters
    // The first is the number of thread blocks
    // The second is the number of threads in each thread block.
    // The product of the first and second is the number of threads that we launch.
    // We need to make sure that the number of blocks * the number of threads is enough
    // to cover the SIZE of the problem we are trying to solve.
    // This is what we have calculated above.
    gpuAddOne<<<NUM_BLOCKS,NUM_THREADS>>>(myGPUArray,SIZE);

    // After executing our kernel, we will copy the results to
    // an array on our CPU.
    // Notice the last parameter where we copy the results.
    // This allows to verify our results on the CPU.
    cudaMemcpy(myCPUArray, myGPUArray, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    // Print out the results of our CPU array after transferring GPU (myGPUArray)
    // memory to the CPU array.
    for(int i=0; i < SIZE; i++){
        printf("myCPUArray [%i]=%i\n",i,myCPUArray[i]);
    }
    // NOTE: This snippet can also be explored next code sample.
    //       This command allocates memory in unified memory and then
    //       returns a pointer that we can keep in 
    //       both host and device code.
    //       This command can make things even easier for us to get started
//    cudaMallocManaged(&myGPUArray,SIZE*sizeof(int));   
//    cudaDeviceSynchronize();

    // We must also free our memory, remember this was allocated on the CPU
    cudaFree(myGPUArray);
    // ====================== GPU Code ====================

    // Free our CPU Memory
    free(myCPUArray);
    return 0;
}
