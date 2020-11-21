/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 4: OpenMP
 * Lesson 6: When Should You Use OpenMP?
 * File: sieve/sieve.serial.c
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

/* Serial code
 *  -- to run, use ./sieve.serial -n N, where N is the value under which to find
 *     primes.
 *  -- see attached module document for discussion of the code and its algorithm
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    /* Declare variables */
    int N = 16; /* The positive integer under which we are finding primes */
    int sqrtN = 4; /* The square root of N, which is stored in a variable to 
                      avoid making excessive calls to sqrt(N) */
    int c = 2; /* Used to check the next number to be circled */
    int m = 3; /* Used to check the next number to be marked */
    int *list; /* The list of numbers -- if list[x] equals 1, then x is marked. 
                  If list[x] equals 0, then x is unmarked. */
    char next_option = ' '; /* Used for parsing command line arguments */
   
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

    /* Allocate memory for list */
    list = (int*)malloc(N * sizeof(int));

    /* Exit if malloc failed */
    if(list == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for list.\n");
        exit(-1);
    }

    /* Run through each number in the list */
    for(c = 2; c <= N-1; c++) {

        /* Set each number as unmarked */
        list[c] = 0;
    }

    /* Run through each number in the list up through sqrtN */
    for(c = 2; c <= sqrtN; c++) {

        /* If the number is unmarked */
        if(list[c] == 0) {

            /* Run through each number bigger than c */
            for(m = c+1; m <= N-1; m++) {

                /* If m is a multiple of c */
                if(m%c == 0) {

                    /* Mark m */
                    list[m] = 1;
                }
            }
        }
    }

    /* Run through each number in the list */
    for(c = 2; c <= N-1; c++) {

        /* If the number is unmarked */
        if(list[c] == 0) {

            /* The number is prime, print it */
            printf("%d ", c);
        }
    }
    printf("\n");

    /* Deallocate memory for list */
    free(list);

    return 0;
}
