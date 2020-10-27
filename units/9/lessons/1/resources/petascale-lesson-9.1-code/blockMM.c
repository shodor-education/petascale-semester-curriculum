/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 9: Optimization
 * Lesson 1: Cache Efficient Matrix Multiplication
 * File: blockMM.c
 * Developed by Paul F. Hemler for the Shodor Education Foundation, Inc.
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

/*
Program performs standard matrix multiply of a square matrix of a 
size specified by the user and block matrix multiply on the same
square matrix.  The squared difference between the two solution
matrices is performed and if there is an error a message is output.
Timing is performed on both ways of multiplying matrices.
*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <time.h>

// Standard matrix multiply for two square matrices A and B.  The 
// product is stored in matrix C.  The function assumes matrices
// A and B have been initialized with values but matrix C has NOT
// been initialized.
void matMult(double **A, double **B, double **C, int n)
{
    for (int r = 0; r < n; ++r)
		for (int c = 0; c < n; ++c)
		{
			C[r][c] = 0.0;
			for (int i = 0; i < n; ++i)
				C[r][c] += A[r][i] * B[i][c];
		}
}

// Define my own minimum function
int myMin(int a, int b) { return a <= b ? a : b; }


// Perform block matrix multiplication between the square matrices A and
// B storing the result in matrix C.  The function uses the specified
// blockSize.  Emperical results indicate that a blockSize of 16 is fine
// if the matrices are greater than 1000 X 1000 elements.
void blockMatMult(double **A, double **B, double **C, int n, int blkSize)
{
    // clear out matrix C
    memset(C[0], 0, n * n * sizeof(double));

    double sum;
    int br, bc, r, c, k;
    // For all row blocks
    for (br = 0; br < n; br += blkSize)
    {
		// For all column blocks
		for (bc = 0; bc < n; bc += blkSize)
		{
			// Fill in a sliver of C
			for (r = 0; r < n; ++r)
			{
				// Multiply the 1 X blockSize sliver (partial row) of
				// A with the blockSize X blockSize block of B and store
				// the results in the 1 X blockSize sliver of C
				for (c = br; c < myMin(br + blkSize, n); ++c)
				{
					sum = 0.0;
					for (k = bc; k < myMin(bc + blkSize, n); ++k)
						sum += A[r][k] * B[k][c];
				C[r][c] += sum;
				}
			}
		}
    }
}


// Simple function to print a 10 X 10 square of a matrix given
// the top-left corner.  Useful to display a small protion of
// a matrix to comapre it to another array.
void printMatSquare(double** mat, int rowOffset, int colOffset)
{
    for (int r = rowOffset; r < rowOffset + 10; ++r)
    {
		for (int c = colOffset; c < colOffset + 10; ++c)
			printf("%.3lf  ", mat[r][c]);
		printf("\n");
    }
}


// Compute and return the sum squared difference between two
// square matrices.  It is used to determine if the serial
// and parallel matrix multiplication functions compute the
// same result.
double squaredDiff(double** a, double** b, int n)
{
    double diff, sqrDiff = 0;
    for (int r = 0; r < n; ++r)
		for (int c = 0; c < n; ++c) {
			diff = a[r][c] - b[r][c];
			sqrDiff += (diff * diff);
		}
    return sqrDiff;
}


// Useful function to compute an elapsed time between two times
struct timespec timespecDifference(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}


// The main driver program to allocate, initialize, multiply, and
// time the results for two square matrices.  The standard serial
// version is first applied and then the block matrix multiplication
// function is also called.  An error check is performed to ensure
// both matrix multiplication functions return the same product matrix.
int main(int argc, char* argv[])
{
    // Two command line arguments are required, which are the square
    // matrix size and the square blickSize.
    if (argc != 3) {
		fprintf(stderr, "USAGE: %s n blkSize\n", argv[0]);
		return -1;
    }

    // Convert the two command line arguments into useful values
    int n = strtol(argv[1], NULL, 10);
    int blkSize = strtol(argv[2], NULL, 10);

    // allocate the matrices, and resolve the row pointers
    double** mat1 = (double**)malloc(n * sizeof(double*));
    mat1[0] = (double*)malloc(n * n * sizeof(double));
    for (int r = 1; r < n; ++r)
	mat1[r] = &(mat1[0][r * n]);

    double** mat2 = (double**)malloc(n * sizeof(double*));
    mat2[0] = (double*)malloc(n * n * sizeof(double));
    for (int r = 0; r < n; ++r)
	mat2[r] = &(mat2[0][r * n]);

    double** mat3 = (double**)malloc(n * sizeof(double*));
    mat3[0] = (double*)malloc(n * n * sizeof(double));
    for (int r = 0; r < n; ++r)
	mat3[r] = &(mat3[0][r * n]);

    double** mat4 = (double**)malloc(n * sizeof(double*));
    mat4[0] = (double*)malloc(n * n * sizeof(double));
    for (int r = 0; r < n; ++r)
	mat4[r] = &(mat4[0][r * n]);

    // initialize the two matrices to be multiplied
    for (int r = 0; r < n; ++r)
		for (int c = 0; c < n; ++c)
		{
			mat1[r][c] = rand() / (double)RAND_MAX;
			mat2[r][c] = rand() / (double)RAND_MAX;
		}


    // Time and report the standard matrix multiplication function
    struct timespec time1, time2, diffTime;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    matMult(mat1, mat2, mat3, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    diffTime = timespecDifference(time1, time2);
    double cpuTimeUsed = ((double)diffTime.tv_sec + (double)diffTime.tv_nsec / 1000000000.0);
    printf ("Standard MM time: %lf sec\n", cpuTimeUsed);

    // Time and report the block matrix multiplication function
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    blockMatMult(mat1, mat2, mat4, n, blkSize);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    diffTime = timespecDifference(time1, time2);
    cpuTimeUsed = ((double)diffTime.tv_sec + (double)diffTime.tv_nsec / 1000000000.0);
    printf ("Block MM time: %lf sec\n", cpuTimeUsed);


    // Make sure both matrix multiplication functions give the
    // same results and tell the user if there is a difference.
    double sqrDiff = squaredDiff(mat3, mat4, n);
    if (sqrDiff > 0.0000001)
	printf("squared difference: %lf\n", sqrDiff);


    // Be kind, rewind
    free(mat1[0]);
    free(mat1);
    free(mat2[0]);
    free(mat2);
    free(mat3[0]);
    free(mat3);
    free(mat4[0]);
    free(mat4);

    // All done
    return 0;
}
