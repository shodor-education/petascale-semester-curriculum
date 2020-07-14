// INSTRUCTIONS HOW TO RUN
//
// Replace program_name with the name of the .cu file
// Compile with: nvcc program_name.cu -o program_name
// Run with    : ./program_name

#include<stdio.h>


#define SIZE 10
/*
 * Function: mult_host
 * ----------------------------
 *   Serially multiplies the values in vector a and b to product
 *
 *   a: vector a
 *   b: vector b
 *   c: vector c
 *   n: size of the vectors 
 *   product: vector to store results
 */
void mult_host(int* a, int* b, int *c, int* product, int n) {
	for (int i = 0; i < n; i++)
	{
		product[i] = a[i] * b[i] *c[i];
	}
}
/*
 * Kernel - multiply vectors 
 * ----------------------------
 *   Each thread multiply the values from vector a, b and c to product
 *	 corresponding to the thread index
 *
 *   a: vector a
 *   b: vector b
 *   b: vector c
 *   product: vector to store results
 */
__global__ void mult_device(int* a, int* b, int *c, int* product, int n) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < n)
		product[thread_id] = a[thread_id] * b[thread_id] * c[thread_id];
}
/*
 * Function: product_vect
 * ----------------------------
 *   multiplies and prints all the elements in vector vect for validation
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
	//print the result
	printf("%d \n", total);
}

/*********************************************
 * main
 *********************************************/
int main(void) {
	//host vectors 
	int *h_a, *h_b, *h_c, *h_product;
	//device vectors
	int* d_a, * d_b, *d_c, *d_product;
	size_t size_vect = SIZE*sizeof(int); /* size of the total vectors necessary to allocate memory */
	
	//allocate memory for the vectors on host (cpu)
	h_a = (int*)malloc(size_vect);
	h_b = (int*)malloc(size_vect);
	h_c = (int*)malloc(size_vect);
	h_product = (int*)malloc(size_vect);

	//allocate memory for the vectors on device (gpu)
	cudaMalloc((void **)&d_a, size_vect);
	cudaMalloc((void **)&d_b, size_vect);
	cudaMalloc((void **)&d_c, size_vect);
	cudaMalloc((void **)&d_product, size_vect);

	//initialize the vectors each with value 1
	for (int i = 0; i < SIZE; i++) {
		h_a[i] = 1;
		h_b[i] = 2;
		h_c[i] = 3;
	}

	//use serial function for vector multition
	mult_host(h_a, h_b, h_c, h_product,SIZE);
	//Verify the result by multiplying all the product, should be 2 * SIZE
	printf("Host product:\n");
	sum_vect(h_product);
	
	//Start CUDA processing
	// Copy vector host values to device
	cudaMemcpy(d_a, h_a, size_vect, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size_vect, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, size_vect, cudaMemcpyHostToDevice);

	//define number of threads
	int threads = 1024;
	//define block size in integer
	int block_size = (int)ceil((float)SIZE / threads);
	//execute the kernel with block size and number of threads
	mult_device << <block_size, 1000 >>> (d_a, d_b, d_c, d_product, SIZE);

	// Copy result back to host
	cudaMemcpy(h_product, d_product, size_vect, cudaMemcpyDeviceToHost);
	
	//Verify the result by multing all the product, should be 2 * SIZE
	printf("Device product:\n");
	sum_vect(h_product);
	
	// Release all device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_product);

	// Release all host memory
	free(h_a);
	free(h_b);
	free(h_c);
	free(h_product);
}