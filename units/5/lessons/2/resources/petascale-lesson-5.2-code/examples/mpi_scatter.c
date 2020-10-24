/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 5: MPI
 * Lesson 2: Collective vs. Point-to-Point Communication
 * mpi_scatter.c
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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utils.h"

int main(int argc, char ** argv)
{
  // Initialize MPI environment
  MPI_Init(&argc, &argv);

  // Make it so we can catch the return code of MPI functions
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  // Get rank and size
  int rank;
  TryMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  int size;
  TryMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  // Declare the variables
  int nums[12];
  int myNums[12];

  // Rank 0 initializes the nums variable
  if (rank == 0)
  {
    // Seed the random number generator
    srandom(time(NULL));

    // Fill the array with random numbers less than 100
    for (int i = 0; i < 12; i++)
    {
      nums[i] = 100 * random() / RAND_MAX;
    }

    // Print the nums array
    printf("(1) before: nums: ");
    for (int i = 0; i < 12; i++)
    {
      printf("%d ", nums[i]);
    }
    printf("\n");
  }

  // Scatter
  TryMPI(MPI_Scatter(nums, // Where in memory is the message that will be 
                           // sent?
                     3, // How many elements of the message will be sent to 
                        // each process?
                     MPI_INT, // What is the datatype of the elements to be
                              // sent?
                     myNums, // Where in memory should the received message 
                             // be stored?
                     3, // How many elements of the message will be received by 
                        // the current process?
                     MPI_INT, // What is the datatype of the elements to be
                              // received?
                     0, // Who is the sender?
                     MPI_COMM_WORLD)); // Which processes are involved in this 
                                       // communication?

  // Print the myNums array after the scatter
  char myNumsStr[3*(2+1)]; // 3 numbers, 2 digits + 1 space character each
  strcpy(myNumsStr, "");
  for (int i = 0; i < 3; i++)
  {
    char numStr[3];
    sprintf(numStr, "%2d ", myNums[i]);
    strcat(myNumsStr, numStr);
  }
  printf("(2) after: rank %d: myNums: %s\n", rank, myNumsStr);

  // Finish up with MPI
  TryMPI(MPI_Finalize());

  return 0;
}

