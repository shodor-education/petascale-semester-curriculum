/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 9: Optimization
 * Lesson 1: Cache Efficient Matrix Multiplication
 * File: matMult.c
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
size specified by the user.  The matrix on the right is then transposed
to put it in column-major order.  A row-major by colum-major multiplication
is then performed.  The squared difference between the two solution
matrices is performed and if there is an error a message is output.
Timing is performed and reported for the standard matrix multiply,
the transpose operation, and the second matrix multiplication.
 */

// Standard includes
#include <stdio.h> 
#include <stdlib.h> 
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


// In place matrix transpose of the matrix passed in
void matTranspose(double **A, int n)
{
    double val;
    for (int r = 0; r < n; ++r)
		for (int c = r + 1; c < n; ++c)
		{
			val = A[r][c];
			A[r][c] = A[c][r];
			A[c][r] = val;
		}
}


// Matrix multication for two square matrices A and B, where B has
// already been transposed.  The rows of A are multiplied by
// the rows of B.  The product is stored in matrix C.  The
// function assumes matrices A and B have been initialized with
// values.  Matrix C has NOT been initialized.
void matMultTranspose(double **A, double **B, double **C, int n)
{
    for (int r = 0; r < n; ++r)
		for (int c = 0; c < n; ++c)
		{
			C[r][c] = 0.0;
			for (int i = 0; i < n; ++i)
				C[r][c] += A[r][i] * B[c][i];
		}
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


// Simple function to print the contents of a square matrix
// in matrix form.  This function is useful for debugging
// small examples.
void printMat(double** mat, int nr, int nc) {
    for (int r = 0; r < nr; ++r)
    {
		for (int c = 0; c < nc; ++c)
			printf ("%.3lf  ", mat[r][c]);
		printf("\n");
    }
}


// Compute and return the sum squared difference between two
// square matrices.  It is used to determine if the serial
// and parallel matrix multiplication functions compute the
// same result.
double squaredDiff(double** a, double** b, int nr, int nc) {
    double diff, sqrDiff = 0.0;
    for (int r = 0; r < nr; ++r)
		for (int c = 0; c < nc; ++c)
		{
			diff = a[r][c] - b[r][c];
			sqrDiff += (diff * diff);
		}
    return sqrDiff;
}



// The main driver program to allocate, initialize, multiply, and
// time the results for two square matrices in serial but one version
// uses the transpose of the matrix on the right so it can be accessed
// in row-major order just like the matrix on the left.
// An error check is performed to ensure both matrix multiplication
// functions return the same product matrix.
int main(int argc, char* argv[])
{
    // One command line argument is required, it is the size of the
    // square arrays to be multiplied together
    if (argc != 2) {
		fprintf(stderr, "USAGE: %s n\n", argv[0]);
		return -1;
    }

    // The code is written to allow non-square matrices
    int nRows, nCols;
    nRows = nCols = strtol(argv[1], NULL, 10);

    // mat3 = mat1 * mat2

    // allocate the matrices, and resolve the row pointers
    double** mat1 = (double**)malloc(nRows * sizeof(double*));
    mat1[0] = (double*)malloc(nRows * nCols * sizeof(double));
    for (int r = 1; r < nRows; ++r)
	mat1[r] = &(mat1[0][r * nCols]);

    double** mat2 = (double**)malloc(nRows * sizeof(double*));
    mat2[0] = (double*)malloc(nRows * nCols * sizeof(double));
    for (int r = 0; r < nRows; ++r)
	mat2[r] = &(mat2[0][r * nCols]);

    double** mat3 = (double**)malloc(nRows * sizeof(double*));
    mat3[0] = (double*)malloc(nRows * nCols * sizeof(double));
    for (int r = 0; r < nRows; ++r)
	mat3[r] = &(mat3[0][r * nCols]);

    double** mat4 = (double**)malloc(nRows * sizeof(double*));
    mat4[0] = (double*)malloc(nRows * nCols * sizeof(double));
    for (int r = 0; r < nRows; ++r)
	mat4[r] = &(mat4[0][r * nCols]);

    // initialize the two matrices to be multiplied
    int cnt = 1;
    for (int r = 0; r < nRows; ++r)
		for (int c = 0; c < nCols; ++c)
		{
    	    mat1[r][c] = cnt++;
			mat2[r][c] = nRows * nCols - cnt + 2;
		}


    // Time and report the standard matrix multiplication function
    struct timespec time1, time2, diffTime;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    matMult(mat1, mat2, mat3, nRows);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    diffTime = timespecDifference(time1, time2);
    double cpuTimeUsed = ((double)diffTime.tv_sec + (double)diffTime.tv_nsec / 1000000000.0);
    printf ("Standard MM time: %lf sec\n", cpuTimeUsed);

    
    // Time the inplace transpose of mat2 
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    matTranspose(mat2, nRows);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    diffTime = timespecDifference(time1, time2);
    cpuTimeUsed = ((double)diffTime.tv_sec + (double)diffTime.tv_nsec / 1000000000.0);
    printf ("Transpose time: %lf sec\n", cpuTimeUsed);


    // Time and report the multiplication by the rows of the transpose
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    matMultTranspose(mat1, mat2, mat4, nRows);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    diffTime = timespecDifference(time1, time2);
    cpuTimeUsed = ((double)diffTime.tv_sec + (double)diffTime.tv_nsec / 1000000000.0);
    printf ("Transpose MM time: %lf sec\n", cpuTimeUsed);


    // Make sure serial and parallel matrix multiplication gave the
    // same results and tell the user if there is a difference.
    double sqrDiff = squaredDiff(mat3, mat4, nRows, nCols);
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
