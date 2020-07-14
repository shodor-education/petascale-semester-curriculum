// INSTRUCTIONS HOW TO RUN
//
// Replace program_name with the name of the .cu file
// Compile with: nvcc program_name.cu -o program_name
// Run with    : ./program_name

#include<stdio.h>



/*
 * Kernel - Parallel reduce max using global memory
 * ----------------------------
 *   Find the max of the vector elements in parallel
 *
 *   in: vector with input values
 *   out: vector with output values
 */
__global__ void parallel_reduce(int *in, int *out) {
	//get the thread id from block and number of threads
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	//get the thread id in the block
	int t_id = threadIdx.x;
	//using for loop break continuously into half (1024->512->256...1), until region with 1 element
	for(int i=blockDim.x/2;i>0;i >>= 1){
		if(t_id<i){
			if ( in[thread_id] < in[thread_id + i])// compare with element from other half
				in[thread_id] =in[thread_id + i]; 
		}
		//we need to synchronize so that all the threads complete the first operations
		__syncthreads();
	}
	//need to write the result from this block to global memory
	if(t_id==0){
		out[blockIdx.x] = in[thread_id];
	}
	
}
/*
 * Kernel - Parallel reduce max with shared memory
 * ----------------------------
 *   Find the max of the vector elements in parallel
 *
 *   in: vector with input values
 *   out: vector with output values
 */
 __global__ void parallel_reduce_shared(int *in, int *out) {
	 //use a shared memory
	 extern __shared__ int shared_data[];
	//get the thread id from block and number of threads
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	//get the thread id in the block
	int t_id = threadIdx.x;
	
	//load data from global mem
	shared_data[t_id]=in[thread_id];
	//synchonize threads to load all data
	__syncthreads();
	//using for loop break continously into half (1024->512->256...1), until region with 1 element
	for(int i=blockDim.x/2;i>0;i >>= 1){
		if(t_id<i){
			if ( shared_data[t_id] < shared_data[t_id + i])// compare with element from other half
					shared_data[t_id] =shared_data[t_id + i]; 
		}
		//we need to synchronize so that all the threads complete the first operations
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
 *   max: to store results
 *   n: maximum size of vector a
 */
 void serial_reduce(int *a, int *max, int n) {
	 //use a serial for loop for max operation
	 *max=a[0];
	 for(int i=1;i<n;i++)
		 if(*max<=a[i])
			 *max = a[i];    
}



/*********************************************
 * main
 *********************************************/
 int main(int argc, char **argv){
	//size of the array 2^20
	int size = 1<<20;

 	//host vectors, h_in will contain the original array, we will use max for result
	int *h_in, max=0;
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
		h_in[i] = i;
	}

	//use serial function for max
	serial_reduce(h_in, &max, size);
	
	//Verify the result by adding all the max, should be size
	printf("Serial results:%d\n",max);
	
	//Start CUDA processing
	// Copy vector host values to device
	cudaMemcpy(d_in, h_in, size_vect, cudaMemcpyHostToDevice);

	//define number of threads
	int threads = 1024;
	//define block size in integer
    int block_size = (int)ceil((float)size / threads);
	
	//execute the kernel with block size and number of threads
	//this is the first phase where the computation results 1024 elements
	parallel_reduce<<<block_size,threads>>>(d_in, d_inter);

	//add the 1024 elements to a single result using 
	parallel_reduce<<<1,block_size>>>(d_inter, d_out);
	// Copy result back to host
	cudaMemcpy(&max, d_out, sizeof(int), cudaMemcpyDeviceToHost);
	
	//Verify the result by adding all the max, should be 2 * size
	printf("Device max global :%d\n",max);

	//using shared memory
	/*
	//execute the kernel with block size and number of threads
	//this is the first phase where the computation results 1024 elements
	parallel_reduce_shared<<<block_size,threads,threads*sizeof(int)>>>(d_in, d_inter);
	//add the 1024 elements to a single result using 
	parallel_reduce_shared<<<1,block_size,block_size*sizeof(int)>>>(d_inter, d_out);
	// Copy result back to host
	cudaMemcpy(&max, d_out, sizeof(int), cudaMemcpyDeviceToHost);
	
	//Verify the result by adding all the max, should be 2 * size
	printf("Device max shared :%d\n",max);
	*/
	// Release all device memory
	cudaFree(d_in);
	cudaFree(d_inter);
	cudaFree(d_out);

	// Release all host memory
    free(h_in);

}
