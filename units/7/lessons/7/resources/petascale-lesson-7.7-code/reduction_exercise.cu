/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 7: CUDA
 * Lesson 7: Parallel Reduce in CUDA
 * File: reduction_exercise.cu
 * Developed by Sanish Rai for the Shodor Education Foundation, Inc.
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
// Replace program_name with the name of the .cu file
// Compile with: nvcc program_name.cu -o program_name
// Run with    : ./program_name

#include<stdio.h>

/*
 * Kernel - Parallel reduce sum using shared memory for exercise 1
 * ----------------------------
 *   Find the sum of the vector elements in parallel
 *
 *   in: vector with input values
 *   out: vector with output values
 */
 __global__ void parallel_reduce_ex1(int *in, int *out) {
    extern __shared__ int shared_data[];
    //get the thread id from block and number of threads
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	//get the thread id in the block
    int t_id = threadIdx.x;
    
    shared_data[t_id]=in[thread_id];
    __syncthreads();
	//using for loop break continuously into half (1024->512->256...1), until region with 1 element
	for( int i=1; i < blockDim.x; i *= 2) {
		if (t_id % (2*i) == 0) {
			shared_data[t_id] += shared_data[t_id + i];
		}
		__syncthreads();
		}
	//need to write the result from this block to global memory
	if(t_id==0){
		out[blockIdx.x] = shared_data[0];
	}
	
}

/*
 * Function - Reduce using serial loops
 * ----------------------------
 *   Use a for loop to add the elements
 *
 *   a: vector a
 *   sum: to store results
 *   n: maximum size of vector a
 */
 void serial_reduce(int *a, int *sum, int n) {
	 //use a serial for loop for addition operation
	for(int i=0;i<n;i++)
		*sum+=a[i];
}



/*********************************************
 * main
 *********************************************/
 int main(int argc, char **argv){
	//size of the array 2^20
	int size = 1<<20;

 	//host vectors, h_in will contain the original array, we will use sum for result
	int *h_in, sum=0;
	//device vectors d_in will contain the original array, we will use d_inter to store intermediate results
	//d_out to store final result 
	int* d_in, * d_inter, * d_out;

	size_t size_vect = size*sizeof(int); /* size of the total vectors necessary to allocate memory */
	h_in = (int*)malloc(size_vect);
	//allocate memory for the vectors on device (gpu)
	cudaMalloc((void **)&d_in, size_vect);
	cudaMalloc((void **)&d_inter, size_vect);
	cudaMalloc((void **)&d_out, sizeof(int));

	//initialize the vectors each with value 1 for simplicity
	for (int i = 0; i < size; i++) {
		h_in[i] = 1;
	}

	//use serial function for sum
	serial_reduce(h_in, &sum, size);
	
	//Verify the result by adding all the sum, should be size
	printf("Serial results:%d\n",sum);
	
	//Start CUDA processing
	// Copy vector host values to device
	cudaMemcpy(d_in, h_in, size_vect, cudaMemcpyHostToDevice);

	//define number of threads
	int threads = 1024;
	//define block size in integer
    int block_size = (int)ceil((float)size / threads);
 
    /* exercise solution*/
    
	//execute the kernel with block size and number of threads
	//this is the first phase where the computation results 1024 elements
    parallel_reduce_ex1<<<block_size,threads,threads*sizeof(int)>>>(d_in, d_inter);
    // Copy result back to host, copy only one element

    //add the 1024 elements to a single result using 
    parallel_reduce_ex1<<<1,block_size,block_size*sizeof(int)>>>(d_inter, d_out);
    
    sum=0;
    cudaMemcpy(&sum, d_out, sizeof(int), cudaMemcpyDeviceToHost);
	//Verify the result by adding all the sum, should be size
	printf("Exercise result :%d\n",sum);
	

	// Release all device memory
	cudaFree(d_in);
	cudaFree(d_inter);
	cudaFree(d_out);

	// Release all host memory
    free(h_in);

}
