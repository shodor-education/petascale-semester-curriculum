/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 3: Using a Cluster
 * Lesson 8: Scaling on a Cluster 2
 * File: mpi_hello.c
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

// Import libraries
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
  // Declare variables
  int mpi_rank;
  int mpi_size;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int processor_name_length;

  // Initialize the MPI execution environment
  MPI_Init(&argc, &argv);

  // Get the rank (unique ID) of the calling process
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // Get the number of processes in the group
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Get the name of the processor on which the calling process is running
  MPI_Get_processor_name(processor_name, &processor_name_length);

  // Say hello
  printf(
    "Hello from process %d out of %d on %s\n",
    mpi_rank,
    processor_name,
    mpi_size
  );

  // Terminate the MPI execution environment
  MPI_Finalize();

  return 0;
}
