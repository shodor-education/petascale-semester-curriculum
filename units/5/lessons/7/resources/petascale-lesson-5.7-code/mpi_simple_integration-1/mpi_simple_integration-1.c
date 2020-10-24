/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 5: MPI
 * Lesson 7: MPI Applications
 * File: mpi_simple_integration-1.c
 * Developed by Tiago Sommer Damasceno for the Shodor Education Foundation, Inc.
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

/*******************************************
A C implementation of Area Under a Curve.

To run:
./Integration [Starting_Range] [Ending_Range] [Number of Segments]
*******************************************/
#include <stdio.h>
#include<stdlib.h>
#include <mpi.h>

/*
	f(long double *x)
		- Computes the value of x squared and returns the value.
*/
long double f(long double *x){
	return (*x)*(*x);
}

/*
	main(int argc, char **argv)
		- Contains basic instructions to find the area under a curve.
		- Default settings:
			o X Squared function
			o X ranged from 0 to 1
			o Curve divided into 1000 segments
*/
int main(int argc, char *argv[])
{
	int i, rank, size;
	MPI_Init (&argc, &argv);      				/* starts MPI */
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);      /* get current process id */
	MPI_Comm_size (MPI_COMM_WORLD, &size);      /* get number of processes */

	long long count, SEGMENTS = 1000*size; // 	SEGMETNS is multiplied by the number of processors 
									 //	 in order to divide the work equally.
	long double x, y, dx, area = 0.0, segmentArea = 0.0, rangeEnd = 1.0, rangeStart = 0.0;

	//Checks for inputs. The program exits if it has less than 3.
	if(argc <= 3 || argc > 4) 
	{
		if(rank == 0)printf("Argument Error - Arguments: %i Expected: 3\n", argc-1);
		MPI_Finalize();
		exit(0);
	}

	rangeStart  = atof(argv[1]); 	  // First  Argument: starting range
	rangeEnd	= atof(argv[2]); 	  // Second Argument: ending range
	SEGMENTS 	= atoi(argv[3])*size; // Third  Argument: Number of segments time number 
									  //of processors, in order to divide work equally.
	// Checks for valid inputs.
	if(rangeEnd <= rangeStart || SEGMENTS < 1)
	{
        if(rank == 0)printf("Argument Error - Invalid Argument\n");
        MPI_Finalize();
        exit(0);
    }
	
	// Computes DeltaX: dx
	dx = (rangeEnd-rangeStart)/(long double)SEGMENTS;

	// Each processor computes a part of the area under a curve based on their rank number.
	for (i = rank*(SEGMENTS/size); i<((rank+1)*(SEGMENTS/size)); i++)
	{
	   long double x = i*dx;
	   y = f(&x);
	   segmentArea += (y*dx);
	}

	//	Adds the segmentArea of each processor into the area variable. 
	//(Reduces segmentArea variables into just the area variable)
	MPI_Reduce(&segmentArea, &area, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if(rank == 0)printf("\n-------------------\n Area is: %Lf\n-------------------\n\n",area);
	MPI_Finalize();
	return 0;
}
