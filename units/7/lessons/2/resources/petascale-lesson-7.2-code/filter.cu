/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 7: CUDA
 * Lesson 2: Image Processing
 * File: filter.cu
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
// Compile with: nvcc ./filter.cu ./ppm.cu -o filter
// Run with    : ./filter
//               You should then output ('display output.ppm')
//               to see your results.

#include <stdio.h>
#include <stdlib.h>
#include "ppm.h"

// Here we have a function that runs on the GPU
// or otherwise, this is our 'device' code.
// This function will take in an array of pixels with
// R, G, and B components and multiply their value
// to create a 'grayscale' filter of an image.
__global__
void make_grayscale(unsigned char* pixel_array, int width, int height){
    // Here we are determining the (x,y) coordinate within
    // the 2D image.
    int x   = threadIdx.x + (blockDim.x*blockIdx.x);
    int y   = threadIdx.y + (blockDim.y*blockIdx.y);
  

    // Check to make sure we do not have more threads
    // than the index of our array
    if(x < width && y < height){
        // We then compute which exact pixel we
        // are located at.
        // 'y' selects the row, and 'x' selects
        // the column. The 'width' is the 'pitch'
        // of the image, and we multiply it by 'y'
        // to select which row in the image we are on.
        int offset = (y*width+x);

        // Our pixels is then made up of 3 color componetns
        // R,G, and B. We multiply by '3' because once we
        // select our pixel, we actually have 3 values per pixel.
        // We then select the R,G, and B values by incrementing
        // by +0, +1, or +2 for R,G, and B.
        unsigned char r = pixel_array[offset*3]; 
        unsigned char g = pixel_array[offset*3+1]; 
        unsigned char b = pixel_array[offset*3+2];

        // At this step we now create our grayscale image.
        // We compute a dot product of the pixels, and apply
        // it to each of our R,G, and B components. Because
        // this is a grayscale image, all of the colors are
        // the same value.
        // NOTE: We could compress our image and only have 1
        //       color channel since all the values are the same.
        pixel_array[offset*3]       = r*.21f + g*.71f + b*.07f;
        pixel_array[offset*3+1]     = r*.21f + g*.71f + b*.07f;
        pixel_array[offset*3+2]     = r*.21f + g*.71f + b*.07f;
    }
}

// Here is another function for fun that 'brightens' the
// actual image
// For fun, you make explore this function as well.
__global__
void make_brighter(unsigned char* pixel_array, int size){
    int id  = threadIdx.x+blockDim.x*blockIdx.x;
 
    // Check to make sure we do not have more threads
    // than the index of our array
    if(id < size){
        pixel_array[id] *= 6; 
    }
}


int main(int argc, char** argv){

    // =================== CPU Code =========================
    // Our CPU must be used to perform I/O (i.e. input and
    // output from the disk). So the first thing we do is
    // load the data from our CPU onto our machine.
    // Our goal here is to load an image in a format that
    // we understand, and retrieve the raw the pixel color
    // information. Then we can manipulate the pixel data
    // to apply a filter for our image.
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
    // We next create a pointer which will point to where our
    // data has been allocated on the GPU.
    unsigned char* gpu_pixels_array; 
    // Now we need to allocate a contigous block of memory
    // on the GPU that our pointer will point to.
    // We do this with the cudaMalloc function.
    cudaMalloc(&gpu_pixels_array,sizeOfDataInBytes);
    // We then copy from our 'host' the data from our CPU into our GPU.
    // Again, what this function is doing, is making a memory transfer of CPU
    // to memory that we have allocated on our GPU.
    cudaMemcpy(gpu_pixels_array,myImage->pixels,sizeOfDataInBytes,cudaMemcpyHostToDevice);      
    //  Here are the parameters
    //    <<<number of thread blocks, number of threads per block>>>
    // So in total, we are launching a total number of threads
    // equal to: "number of thread blocks" * "number of threads per block"        
    // NOTE:    Be careful here with the first parameter of which array you are
    //          passing as the first parameter. Remember that CUDA functions
    //          that are executing on a 'device' should only address pointers
    //          and memory locations that are on the actual device, thus we
    //          pass in our gpu_pixels_array where we copied all of our pixels
    //          to in the previous step.
    dim3 dimGrid(myImage->width*3,myImage->height*3,1);
    dim3 dimBlock(1,1,1);
    // Now we call our grayscale function
    // Note that we are passing in our gpu_pixels_array in, as when we
    // work with GPU kernel code, we need to work with GPU memory
    // (i.e. blocks of memory allocated with cudaMalloc).
    make_grayscale<<<dimGrid,dimBlock>>>(gpu_pixels_array,myImage->width,myImage->height);
    // After our kernel has been called, we copy the data from our GPU memory
    // back onto our CPU memory.
    // Our goal is to get the memory back on the CPU, so we can use
    // CPU functions to write our file back to disk.
    cudaMemcpy(myImage->pixels,gpu_pixels_array,sizeOfDataInBytes,cudaMemcpyDeviceToHost);     
    // Free memory that we have allocated on the GPU.
    cudaFree(gpu_pixels_array);
    // =================== GPU Code =========================
    
     
    // =================== CPU Code =========================
    // Write our file back to disk
    const char* output = "./output.ppm";
    savePPMImage(output, myImage);

    // Free memory that we have allocated on the CPU.
    free(myImage);
    // =================== CPU Code =========================
    

    return 0;
}
