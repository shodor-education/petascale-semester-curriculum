/***************************************************************
 * File Name: vect_add.c
 * Author: Mobeen Ludin
 * Date: March 29, 2016
 * 
 * How to Compile:
 * By Default the make file uses pgi compiler
 *  $ make clean && make vect_add  
 * How to compile with cray compiler
 *  $ cc -h pragma=acc vect_add.c -o vect_add.exe
 * How to Run: $ aprun ./vect_add.exe
 **************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define VEC_SIZE 10000 // How big is the vector going to be.

// Function declaration
double* vector_add(double *vectA, double *vectB, double *vectSum);

int main(int argc, char *argv[]){
	// Timers
    struct timeval start_time, stop_time, elapsed_time;
    // Declare pointer variables for vector arrays: vectA, vectB and vectSum
    double *vectA;      
    double *vectB;      
    double *vectSum;
    
	// Allocate enough memory to hold each vector array
    vectA = malloc(sizeof(double) * VEC_SIZE);   
    vectB = malloc(sizeof(double) * VEC_SIZE);   
    vectSum = malloc(sizeof(double) * VEC_SIZE); 
    
    int i;  // Declaring loop index variable
    
    //Timer starts here
    gettimeofday(&start_time,NULL);

    // Call function to carryout vector addition
	vector_add(vectA, vectB, vectSum);
    
	// Stop timer
    gettimeofday(&stop_time,NULL);
    timersub(&stop_time, &start_time, &elapsed_time);

    printf("Total time was %f seconds.\n",
            elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
    
	// Print last 10 rows
    for(i = VEC_SIZE-10; i < VEC_SIZE; i++){
        printf("  Vector Add: A[ %.2f ] + B[ %.2f ] = Sum[ %.2f ] \n", 
                vectA[i], vectB[i], vectSum[i]);
    }

    free(vectA);    // Free up the memory allocated earlier
    free(vectB);    // Take back memory to use for something else
    free(vectSum);  // If dont reclaim, that memory will still be occupied.

} // End: main()

double* vector_add(double *vectorA, double *vectorB, double *vectorSum){
    int i, done;
#pragma acc data copyin(vectorA[0:VEC_SIZE], vectorB[0:VEC_SIZE]) copyout(vectorSum[0:VEC_SIZE])
while ( !done && i < VEC_SIZE)
{

    //#pragma acc parallel copyin(vectorA[0:VEC_SIZE], vectorB[0:VEC_SIZE]), copyout(vectorSum[0:VEC_SIZE])
    #pragma acc parallel 
    { 
        #pragma acc loop
        for (i = 0; i < VEC_SIZE; i++){ 
            vectorA[i] = (i+1)*10;      //initialize array elements
            vectorB[i] = (i+1)*20;      //do same for array vectorB
        }
    }
    #pragma acc parallel 
    { 
        #pragma acc loop
        for ( i = 0; i < VEC_SIZE; i++){
            vectorSum[i] = vectorA[i] + vectorB[i];
        }
    }
    done = 1;
}
    return(vectorSum); 
}
