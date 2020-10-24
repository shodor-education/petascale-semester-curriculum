/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 4: OpenMP
 * Lesson 8: Markov chains, Matrix multiply
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
Program performs both the standard (serial) and parallel (OpenMP)
matrix multiplication for square matrices.  Optional command line
arguments control the size of matrix and the number of threads 
that should be used for the parallel version.  Timing information
is recorded and reported when the program terminates.

There is an accompaning BASH shell script, which was used to 
generate all the results presented.
*/

// Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // needed for memset(...)
#include <omp.h>


// Simple function to print the contents of a square matrix
// in matrix form.  This function is useful for debugging
// small examples.
void printMat(double** M, int n)
{
    for (int r = 0; r < n; ++r)
    {
		for (int c = 0; c < n; ++c)
			printf("%.3lf  ", M[r][c]);
		printf("\n");
    }
}

// Compute and return the sum squared difference between two
// square matrices.  It is used to determine if the serial
// and parallel matrix multiplication functions compute the
// same result.
double squareDifference(double** A, double** B, int n)
{
    double error = 0.0;
    for (int r = 0; r < n; ++r)
		for (int c = 0; c < n; ++c)
			error += (A[r][c] - B[r][c]) * (A[r][c] - B[r][c]);
    return error;
}

// Perform standard (serial) matrix multiplication between
// two square matrices, A and B, C = AB
void matMult(double **A, double **B, double **C, int n)
{
    for (int r = 0; r < n; r++)
		for (int c = 0; c < n; c++)
			for (int k = 0; k < n; k++)
				C[r][c] += A[r][k] * B[k][c];
}


// Perform parallel (OpenMP)  matrix multiplication between
// two square matrices, A and B, C = AB.  This function demonstrates
// the simplicity of converting serial code into parallel code.
void matMultParallel(double **A, double **B, double **C, int n)
{
#pragma omp parallel for shared(A, B, C, n)
	for (int r = 0; r < n; r++)
	    for (int c = 0; c < n; c++)
			for (int k = 0; k < n; k++)
				C[r][c] += A[r][k] * B[k][c];
}



// The main driver program to allocate, initialize, multiply, and
// time the results for  two square matrices in serial and parallel.
// An error check is performed to ensure both matrix multiplication
// functions return the same product matrix.
int main(int argc, char* argv[])
{
    // Two command line arguments are optional.  Reasonable
    // default values are supplied.
    int n, nThreads;
    switch (argc)
    {
	case 1:
		n = 10;
		nThreads = 1;
		break;
    case 2:
		n = atoi(argv[1]);
		nThreads = 4;
		break;
    case 3:
		n = atoi(argv[1]);
		nThreads = atoi(argv[2]);
		break;
    default:
		printf(stderr, "USAGE: %s [n] [nThreads]\n", argv[0]);
		exit(-1);
    }

    // Variables for matrix intital values
    double nSquared = n * n;
    double twoN = 2.0 * n;

    // Tell OpenMP how many threads to spawn for parallel code sections
    omp_set_num_threads(nThreads);

    // allocate the matrices
    double **A, **B, **C, **D;
    A = (double**)malloc(n * sizeof(double*));
    A[0] = (double*)malloc(n * n * sizeof(double));

    B = (double**)malloc(n * sizeof(double*));
    B[0] = (double*)malloc(n * n * sizeof(double));

    C = (double**)malloc(n * sizeof(double*));
    C[0] = (double*)malloc(n * n * sizeof(double));

    D = (double**)malloc(n * sizeof(double*));
    D[0] = (double*)malloc(n * n * sizeof(double));

#pragma omp parallel shared(A, B, C, D)
    {
	// Resolve all row pointers in parallel and initialize to 0
#pragma omp for
	for (int r = 0; r < n; ++r)
	{
	    A[r] = &A[0][r * n];
	    memset(A[r], 0, n * sizeof(double));
	    B[r] = &B[0][r * n];
	    memset(B[r], 0, n * sizeof(double));
	    C[r] = &C[0][r * n];
	    memset(C[r], 0, n * sizeof(double));
	    D[r] = &D[0][r * n];
	    memset(D[r], 0, n * sizeof(double));
	}

	// initialize matrices A and B with some fractonal values
#pragma omp for
	for (int r = 0; r < n; ++r)
	    for (int c = 0; c < n; ++c)
	    {
			A[r][c] = (r + c) / twoN;
			B[r][c] = r * c / nSquared;
	    }
    } // end parallel

    // Time and report the serial matrix multiplication function
    double time1, time2;
    time1 = omp_get_wtime();
    matMult(A, B, C, n);
    time2 = omp_get_wtime();
    printf("serial MM: %.3lf\n", time2 - time1);

    // Time and report the parallel matrix multiplication function
    time1 = omp_get_wtime();
    matMultParallel(A, B, D, n);
    time2 = omp_get_wtime();
    printf("parallel MM: %.3lf\n", time2 - time1);

    // Make sure serial and parallel matrix multiplication gave the
    // same results and tell the user is there was a difference.
    double sqrDiff = squareDifference(C, D, n);
    if (sqrDiff > 0.0000001)
		fprintf(stderr, "Matrix multiplication error\n");

    // Be kind, rewind
    free(A[0]);
    free(A);
    free(B[0]);
    free(B);
    free(C[0]);
    free(C);
    free(D[0]);
    free(D);

    return 0;
}
