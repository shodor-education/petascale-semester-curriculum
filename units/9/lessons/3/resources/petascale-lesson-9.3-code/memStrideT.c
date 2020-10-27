/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 9: Optimization
 * Lesson 3: Cache Memory Efficiency
 * File: memStrideT.c
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
Program sequentially accesses elements in an array of a user specified 
number of Mega elements (chars, ints, floats, doubles, ...).  A second
parameter specifies the number of elements to increment over before the
next access.  1 means sequential access, 2 means every other element is
accessed and then the skipped elements are accessed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>   /* clock() */
#include <sys/time.h>

// Macros for consistency
#define KILO 1024
#define MEGA (KILO*KILO)
#define GIGA (KILO*MEGA)

// Use myDataType throughout the code
// Change myDataType to stride through different data types
typedef unsigned char myDataType;

int numPasses = 10;

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



// The main driver program to allocate a user specified large linear
// array to exercise different memory access patterns.  The results
// are timed and reported.
// The program requires two command line arguments. The first is the
// number of Mebi elements in the linear array and the second is the
// number of elements in the stride.
int main (int argc, char *argv[])
{
    // Check usage
    if (argc != 3)
    {
		fprintf (stderr, "USAGE: %s nMB stride\n", argv[0]);
		exit(-1);
    }

    // Convert the first command line argument into the number of Mebi
    // elements for the lineas array
    int nMB = strtol(argv[1], NULL, 10);
    nMB *= MEGA;

    // Convert the second command line argument into the stride, or number
    // of elements away from the current element the next element is
    int stride = strtol(argv[2], NULL, 10);

    // Allocate heap space of the given size and verify its existance
    myDataType *buffer = (myDataType*)malloc(nMB * sizeof(myDataType));
    if (buffer == NULL)
    {
		fprintf (stderr, "ERROR: malloc failed\n");
		exit(-1);
    }
	
    // The value to be written into each element in the linear array
    // Without changing the array the compiler optimizes the loops out!
    myDataType val = 10;
    
    // Make multiple passes over each element in the array, skipping some
    // as you go.  The skipped elements get written in a subsequent run
    // through array.  Tme the results.
    struct timespec time1, time2, diffTime;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    for (int pass = 0; pass < numPasses; ++pass)
    {
		for (int j = 0; j < stride; ++j)
		{
			for (int i = j; i < nMB; i += stride)
			{
				buffer[i] = val;
			}
		}
    }     
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    diffTime = timespecDifference(time1, time2);
    double cpuTimeUsed = ((double)diffTime.tv_sec + (double)diffTime.tv_nsec / 1000000000.0);
    cpuTimeUsed /= (double)numPasses;

    // Output the reuslts in as comma seperated values
    printf ("%d, %.3lf\n", stride, cpuTimeUsed);
    
    // Be nice, rewind
    free(buffer);

    // All done
    return 0;
}
