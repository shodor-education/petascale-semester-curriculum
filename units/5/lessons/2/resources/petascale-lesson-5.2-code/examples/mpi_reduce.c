/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 5: MPI
 * Lesson 2: Collective vs. Point-to-Point Communication
 * mpi_reduce.c
 * Developed by Aaron Weeden and Mobeen Ludin for the Shodor Education
 * Foundation, Inc.
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
#include "utils.h"

int main(int argc, char ** argv)
{
  // Initialize MPI environment
  MPI_Init(&argc, &argv);

  // Make it so we can catch the return code of MPI functions
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  // Get rank and size
  int rank;
  TryMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  int size;
  TryMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  // Initialize the variables
  int mine = rank+1;
  int total = 0;

  // Print the variables before the reduction
  printf("(1) before: rank %d: mine: %d, total: %d\n", rank, mine, total);

  // Reduce
  TryMPI(MPI_Reduce(&mine, // Where in memory is the message that will be sent?
                    &total, // Where in memory should the received message be stored?
                    1, // How many elements are in the message?
                    MPI_INT, // What is the datatype of the elements in the message?
                    MPI_SUM, // What is the operation to be performed? 
                    0, // Who is the receiver?
                    MPI_COMM_WORLD)); // Which processes are involved in this communication?

  // Print the variables before the reduction
  printf("(2) after: rank %d: mine: %d, total: %d\n", rank, mine, total);

  // Finish up with MPI
  TryMPI(MPI_Finalize());

  return 0;
}

