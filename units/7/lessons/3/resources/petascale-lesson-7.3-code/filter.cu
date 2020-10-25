// INSTRUCTIONS HOW TO RUN
//
// Compile with: nvcc ./filter.cu ./ppm.cu -o filter
// Run with    : ./filter

#include <stdio.h>
#include <stdlib.h>
#include "ppm.h"


// Some helper functions found in "ppm.h"
extern ppm_t* loadPPMImage(const char* filepath);
extern void savePPMImage(const char* filename, ppm_t* ppm);

// Here we have a function that runs on the GPU
// or otherwise, this is our 'device' code.
__global__
void make_grayscale(unsigned char* pixel_array, int size){
    int id  = threadIdx.x+blockDim.x*blockIdx.x;
  
    // Check to make sure we do not have more threads
    // than the index of our array
    if(id < size){
        pixel_array[id] *= 6; 
    }
}


int main(int argc, char** argv){

    // =================== CPU Code =========================
    // Our goal here is to load an image, and get the pixels
    // from the image into our format. Then we can manipulate
    // these pixel values.
    const char* input = "./safari.ppm"; 
    ppm_t* myImage = loadPPMImage(input);
    // =================== CPU Code =========================

    // =================== GPU Code =========================
    
    
    // Here we are going to launch our kernel:
    // Now let's allocate some memory in CUDA for our pixels array
    // We need to allocate enough memory for all of our pixels, as well
    // as each of the R,G,B color components
    int numberOfPixels = myImage->width * myImage->height;
    int sizeOfDataInBytes = numberOfPixels*3*sizeof(unsigned char);
    unsigned char* gpu_pixels_array = (unsigned char*)malloc(sizeOfDataInBytes); 
    // Now we need to allocate some memory on the GPU.
    // We do this with the cudaMalloc function.
    cudaMalloc(&gpu_pixels_array,sizeOfDataInBytes);
    // We then copy from our 'host' the data from our CPU into our GPU.
    // Again, what this function is doing, is making a memory transfer of CPU
    // to memory that we have allocated on our GPU.
    cudaMemcpy(gpu_pixels_array,myImage->pixels,sizeOfDataInBytes,cudaMemcpyHostToDevice);     
 
    
    //  Here are the parameters
    //    <<<number of thread blocks, number of threads>>>
    // So in total, we are launching a total number of threads
    // equal to: "number of thread blocks" * "number of threads"        
    // NOTE:    Be careful here with the first parameter of which array you are
    //          passing as the first parameter. Remember that CUDA functions
    //          that are executing on a 'device' should only address pointers
    //          and memory locations that are on the actual device, thus we
    //          pass in our gpu_pixels_array where we copied all of our pixels
    //          to in the previous step.
    make_grayscale<<<ceil(sizeOfDataInBytes/256.0),256>>>(gpu_pixels_array,sizeOfDataInBytes);


    cudaMemcpy(myImage->pixels,gpu_pixels_array,sizeOfDataInBytes,cudaMemcpyDeviceToHost);     
    // Store the pixels from our GPU back into our CPU
//    myImage->pixels = gpu_pixels_array;
    // =================== GPU Code =========================
    
    
    
    // =================== CPU Code =========================
    const char* output = "./output.ppm";
    savePPMImage(output, myImage);

    // Free memory that we have allocated on the CPU.
    free(myImage);
    // =================== CPU Code =========================
    
    // Free memory that we have allocated on the GPU.
    cudaFree(gpu_pixels_array);

    return 0;
}
