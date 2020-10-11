/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 3: Using a Cluster
 * Lesson 5: Running Code on a Cluster 1
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
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>

void TryMPI(int const err);

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  TryMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  int size;
  TryMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  char name[80];
  int len;
  MPI_Get_processor_name(name, &len);
  printf("PE %2d is on core %d on node %s\n", rank, sched_getcpu(), name);
  TryMPI(MPI_Finalize());
  return 0;
}

void TryMPI(int const err)
{
  if (err != MPI_SUCCESS)
  {
    char string[120];
    int resultlen;
    MPI_Error_string(err, string, &resultlen);
    fprintf(stderr, "ERROR: MPI: %s\n", string);
    MPI_Abort(MPI_COMM_WORLD, err);
  }
}

