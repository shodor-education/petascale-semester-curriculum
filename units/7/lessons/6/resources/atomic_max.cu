// INSTRUCTIONS HOW TO RUN
//
// Replace program_name with the name of the .cu file
// Compile with: nvcc program_name.cu -o program_name
// Run with    : ./program_name

#include<stdio.h>


#define SIZE 900000

/*
 * Kernel - Find max without atomic
 * ----------------------------
 *   Each thread checks if max is greater than its value,
 *   if yes then replaces the max
 *
 *   a: vector a
 *   max: to store results
 *   n: maximum size of vector a
 */
__global__ void simple_max(int *a, int *max, int n) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < n && a[thread_id]>*max)
		*max =a[thread_id];
}

/*
 * Kernel - Find max with atomic
 * ----------------------------
 *   Each thread checks if max is greater than its value,
 *   if yes then replaces the max using atomicMax
 *
 *   a: vector a
 *   max:  to store results
 *   n: maximum size of vector a

 */
__global__ void atomic_max(int *a, int *max, int n) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < n)
		atomicMax(max,a[thread_id]);
}


/*********************************************
 * main
 *********************************************/
int main(void) {
	//host variables 
	int *h_var, *h_max ;
	int max=0;

	//device variables
	int* d_var, *d_max;
	
	size_t size_vect = SIZE*sizeof(int); /* size of the total vectors necessary to allocate memory */
	
	//allocate memory for the variables on host (cpu)
	h_var = (int*)malloc(size_vect);
	h_max = (int*)malloc(sizeof(int));
	h_max=&max;/* h_max is to store the max on the host device */
	
	//allocate memory for the variables on device (gpu)
	cudaMalloc((void **)&d_var, size_vect);
	cudaMalloc((void **)&d_max, size_vect);
	cudaMemset ((void **)d_max,0, sizeof(int));
	
	//initialize the vectors each with value 1
	for (int i = 0; i < SIZE; i++) {
		h_var[i] = i+1;
	}

	//Start CUDA processing
	// Copy host values to device
	cudaMemcpy(d_var, h_var, size_vect, cudaMemcpyHostToDevice);

	//define number of threads
	int threads = 1024;
	//define block size in integer
	int block_size = (int)ceil((float)SIZE / threads);
	
	//execute the kernel with block size and number of threads
	simple_max << <block_size, threads >>> (d_var, d_max, SIZE);

	// Copy result back to host
	cudaMemcpy(h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
	
	//Verify the result, should be equal to SIZE
	printf("Result without atomic add : %d\n",max);

	//reset d_max to 0
	cudaMemset ((void **)d_max,0, sizeof(int));
	
	//execute the kernel with block size and number of threads
	atomic_max << <block_size, threads >>> (d_var, d_max, SIZE);

	// Copy result back to host
	cudaMemcpy(h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
	
	//Verify the result, should be equal to SIZE
	printf("Result using atomic add : %d\n",max);
	
	// Release all device memory
	cudaFree(d_var);

	// Release all host memory
	free(h_var);
}