/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 7: CUDA
 * Lesson 1: Introduction to CUDA GPGPU
 * File: hello_cpu.c
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

// LESSON INTRODUCTION
// 
// INSTRUCTIONS HOW TO RUN
//
// Compile with: gcc hello_cpu.cu -o hello_cpu
// Run with    : ./hello_cpu


// Include our C Standard Libraries
#include <stdio.h>
#include <stdlib.h>

// This is our problem size
#define SIZE (640*480)

// Increments all values in array by 1
void cpuAddOne(int* array,int size){
    for(unsigned int i=0; i < size; i++){
        array[i]+=1;
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
    // (For bonus points)
    // Alternatively you can set each byte to a specific value (see 'memset')
     
    // Now let's actually solve a problem by incrementing our array.
    cpuAddOne(myCPUArray,SIZE);
    
    // Output and verify our result.
    // Our expected result is that each value in our
    // array will have been incremented by 1 from its
    // initial value.
    // Verifying our Output on the CPU code will also help us make sure
    // that our GPU code is also correct.
    for(int i=0; i < SIZE; i++){
        printf("myCPUArray[%i]=%i\n",i,myCPUArray[i]);
    }

    // An aside: Is incrementing every value in an array really necessary
    //           or is this just a toy problem?
    // Answer:   This actually can be a useful function--later on we are 
    //           going to implement an 'image filter' where we want to
    //           apply the same function to many values at once, and we will
    //           want to learn how to do this as fast as possible! This is
    //           where we will learn GPU programming can excel!
   
    // When we are done with our array, we free (i.e. reclaim) our memory.
    // We do this in our programs so that we can reuse the memory, because
    // remember, the memory on our machines is finite!
    free(myCPUArray);

    return 0;
}
