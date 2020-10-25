/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 7: CUDA
 * Lesson 5: Adding 2 vectors in CUDA
 * File: vector_addition.cu
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

#include<stdio.h>


#define SIZE 1000000
/*
 * Function: add_host
 * ----------------------------
 *   Serially adds the values in vector a and b to sum
 *
 *   a: vector a
 *   b: vector b
 *   n: size of the vectors 
 *   sum: vector to store results
 */
void add_host(int* a, int* b, int* sum, int n) {
	for (int i = 0; i < n; i++)
	{
		sum[i] = a[i] + b[i];
	}
}
/*
 * Kernel - Add vectors 
 * ----------------------------
 *   Each thread adds the values from vector a and b to sum
 *	 corresponding to the thread index
 *
 *   a: vector a
 *   b: vector b
 *   sum: vector to store results
 */
__global__ void add_device(int* a, int* b, int* sum, int n) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < n)
		sum[thread_id] = a[thread_id] + b[thread_id];
}
/*
 * Function: sum_vect
 * ----------------------------
 *   Adds and prints all the elements in vector vect for validation
 *
 *   vect: vector
 */
void sum_vect(int* vect)
{
	int total = 0;
	//sum all the elements 
	for (int i = 0; i < SIZE; i++) 
	{ 
		total += vect[i];
	}
	//print results
	printf("%d \n", total);
}

/*********************************************
 * main
 *********************************************/
int main(void) {
	//host vectors 
	int *h_a, *h_b, *h_sum;
	//device vectors
	int* d_a, * d_b, * d_sum;
	size_t size_vect = SIZE*sizeof(int); /* size of the total vectors necessary to allocate memory */
	
	//allocate memory for the vectors on host (cpu)
	h_a = (int*)malloc(size_vect);
	h_b = (int*)malloc(size_vect);
	h_sum = (int*)malloc(size_vect);

	//allocate memory for the vectors on device (gpu)
	cudaMalloc((void **)&d_a, size_vect);
	cudaMalloc((void **)&d_b, size_vect);
	cudaMalloc((void **)&d_sum, size_vect);

	//initialize the vectors each with value 1
	for (int i = 0; i < SIZE; i++) {
		h_a[i] = 1;
		h_b[i] = 1;
	}

	//use serial function for vector addition
	add_host(h_a, h_b, h_sum,SIZE);
	//Verify the result by adding all the sum, should be 2 * SIZE
	printf("Host sum:\n");
	sum_vect(h_sum);
	
	//Start CUDA processing
	// Copy vector host values to device
	cudaMemcpy(d_a, h_a, size_vect, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size_vect, cudaMemcpyHostToDevice);

	//define number of threads
	int threads = 1024;
	//define block size in integer
	int block_size = (int)ceil((float)SIZE / threads);
	//execute the kernel with block size and number of threads
	add_device << <block_size, threads >>> (d_a, d_b, d_sum, SIZE);

	// Copy result back to host
	cudaMemcpy(h_sum, d_sum, size_vect, cudaMemcpyDeviceToHost);
	
	//Verify the result by adding all the sum, should be 2 * SIZE
	printf("Device sum:\n");
	sum_vect(h_sum);
	
	// Release all device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_sum);

	// Release all host memory
	free(h_a);
	free(h_b);
	free(h_sum);
}
