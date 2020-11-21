/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 4: OpenMP
 * Lesson 6: When Should You Use OpenMP?
 * File: sieve/sieve.mpi.c
 * Developed by Aaron Weeden for the Shodor Education Foundation, Inc.
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

/* MPI code
 *  -- to run, use mpirun -np p ./sieve.serial -n N, where p is the number of
 *     processes and N is the value under which to find primes.
 *  -- see attached module document for discussion of the code and its algorithm
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    /* Declare variables */
    int N = 16; /* The positive integer under which we are finding primes */
    int sqrtN = 0; /* The square root of N, which is stored in a variable to 
                      avoid making excessive calls to sqrt(N) */
    int c = 0; /* Used to check the next number to be circled */
    int m = 0; /* Used to check the next number to be marked */
    int *list1; /* The list of numbers <= sqrtN -- if list1[x] equals 1, then x 
                   is marked.  If list1[x] equals 0, then x is unmarked. */
    int *list2; /* The list of numbers > sqrtN -- if list2[x-L] is marked, then 
                   x is marked.  If list2[x-L] equals 0, then x is unmarked. */
    char next_option = ' '; /* Used for parsing command line arguments */
    int S = 0; /* A near-as-possible even split of the count of numbers above 
                  sqrtN */
    int R = 0; /* The remainder of the near-as-possible even split */
    int L = 0; /* The lowest number in the current process's split */
    int H = 0; /* The highest number in the current process's split */
    int r = 0; /* The rank of the current process */
    int p = 0; /* The total number of processes */

    /* Initialize the MPI Environment */
    MPI_Init(&argc, &argv);

    /* Determine the rank of the current process and the number of processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
   
    /* Parse command line arguments -- enter 'man 3 getopt' on a shell to see
       how this works */
    while((next_option = getopt(argc, argv, "n:")) != -1) {
        switch(next_option) {
            case 'n':
                N = atoi(optarg);
                break;
            case '?':
            default:
                fprintf(stderr, "Usage: %s [-n N]\n", argv[0]);
                exit(-1);
        }
    }

    /* Calculate sqrtN */
    sqrtN = (int)sqrt(N);

    /* Calculate S, R, L, and H */
    S = (N-(sqrtN+1)) / p;
    R = (N-(sqrtN+1)) % p;
    L = sqrtN + r*S + 1;
    H = L+S-1;
    if(r == p-1) {
        H += R;
    }

    /* Allocate memory for lists */
    list1 = (int*)malloc((sqrtN+1) * sizeof(int));
    list2 = (int*)malloc((H-L+1) * sizeof(int));

    /* Exit if malloc failed */
    if(list1 == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for list1.\n");
        exit(-1);
    }
    if(list2 == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for list2.\n");
        exit(-1);
    }

    /* Run through each number in list1 */
    for(c = 2; c <= sqrtN; c++) {

        /* Set each number as unmarked */
        list1[c] = 0;
    }
    
    /* Run through each number in list2 */
    for(c = L; c <= H; c++) {

        /* Set each number as unmarked */
        list2[c-L] = 0;
    }

    /* Run through each number in list1 */
    for(c = 2; c <= sqrtN; c++) {

        /* If the number is unmarked */
        if(list1[c] == 0) {

            /* Run through each number bigger than c in list1 */
            for(m = c+1; m <= sqrtN; m++) {

                /* If m is a multiple of c */
                if(m%c == 0) {

                    /* Mark m */
                    list1[m] = 1;
                }
            }

            /* Run through each number bigger than c in list2 */
            for(m = L; m <= H; m++)
            {
                /* If m is a multiple of C */
                if(m%c == 0)
                {
                    /* Mark m */
                    list2[m-L] = 1;
                }
            }
        }
    }

    /* If Rank 0 is the current process */
    if(r == 0) {

        /* Run through each of the numbers in list1 */
        for(c = 2; c <= sqrtN; c++) {

            /* If the number is unmarked */
            if(list1[c] == 0) {

                /* The number is prime, print it */
                printf("%d ", c);
            }
        }

        /* Run through each of the numbers in list2 */
        for(c = L; c <= H; c++) {

            /* If the number is unmarked */
            if(list2[c-L] == 0) {

                /* The number is prime, print it */
                printf("%d ", c);
            }
        }

        /* Run through each of the other processes */
        for(r = 1; r <= p-1; r++) {
            
            /* Calculate L and H for r */
            L = sqrtN + r*S + 1;
            H = L+S-1;
            if(r == p-1) {
                H += R;
            }
            
            /* Receive list2 from the process */
            MPI_Recv(list2, H-L+1, MPI_INT, r, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);

            /* Run through the list2 that was just received */
            for(c = L; c <= H; c++) {

                /* If the number is unmarked */
                if(list2[c-L] == 0) {

                    /* The number is prime, print it */
                    printf("%d ", c);
                }
            }
        }
        printf("\n");

        /* If the process is not Rank 0 */
    } else {

        /* Send list2 to Rank 0 */
        MPI_Send(list2, H-L+1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    /* Deallocate memory for list */
    free(list2);
    free(list1);

    /* Finalize the MPI environment */
    MPI_Finalize();

    return 0;
}
