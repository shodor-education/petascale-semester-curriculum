// INSTRUCTIONS HOW TO RUN
//
// Replace program_name with the name of the .cu file
// Compile with: nvcc program_name.cu -o program_name
// Run with    : ./program_name

#include<stdio.h>


#define SIZE 900000

/*
 * Kernel - Count threads without atomic
 * ----------------------------
 *   Each thread adds its value to sum
 *
 *   a: vector a
 *   sum: to store results
 *   n: maximum size of vector a
 */
__global__ void simple_count(int *a, int *sum, int n) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < n)
		*sum =*sum+a[thread_id];
}

/*
 * Kernel - Count threads with atomic
 * ----------------------------
 *   Each thread it's values sum using atomicAdd
 *
 *   a: vector a
 *   sum:  to store results
 *   n: maximum size of vector a

 */
__global__ void atomic_count(int *a, int *sum, int n) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < n)
		atomicAdd(sum,a[thread_id]);
}


/*********************************************
 * main
 *********************************************/
int main(void) {
	//host variables 
	int *h_var, *h_sum ;
	int sum=0;

	//device variables
	int* d_var, *d_sum;
	
	size_t size_vect = SIZE*sizeof(int); /* size of the total vectors necessary to allocate memory */
	
	//allocate memory for the variables on host (cpu)
	h_var = (int*)malloc(size_vect);
	h_sum = (int*)malloc(sizeof(int));
	h_sum=&sum;/* h_sum is to store the sum on the host device */
	
	//allocate memory for the variables on device (gpu)
	cudaMalloc((void **)&d_var, size_vect);
	cudaMalloc((void **)&d_sum, size_vect);
	cudaMemset ((void **)d_sum,0, sizeof(int));
	
	//initialize the vectors each with value 1
	for (int i = 0; i < SIZE; i++) {
		h_var[i] = 1;
	}

	//Start CUDA processing
	// Copy host values to device
	cudaMemcpy(d_var, h_var, size_vect, cudaMemcpyHostToDevice);

	//define number of threads
	int threads = 1024;
	//define block size in integer
	int block_size = (int)ceil((float)SIZE / threads);
	
	//execute the kernel with block size and number of threads
	simple_count << <block_size, threads >>> (d_var, d_sum, SIZE);

	// Copy result back to host
	cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
	
	//Verify the result, should be equal to SIZE
	printf("Result without atomic add : %d\n",sum);

	//reset d_sum to 0
	cudaMemset ((void **)d_sum,0, sizeof(int));
	
	//execute the kernel with block size and number of threads
	atomic_count << <block_size, threads >>> (d_var, d_sum, SIZE);

	// Copy result back to host
	cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
	
	//Verify the result, should be equal to SIZE
	printf("Result using atomic add : %d\n",sum);
	
	// Release all device memory
	cudaFree(d_var);

	// Release all host memory
	free(h_var);
}