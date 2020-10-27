/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 5: MPI
 * Lesson 2: Collective vs. Point-to-Point Communication
 * mpi_pi_montecarlo.c
 * Developed by Mobeen Ludin for the Shodor Education Foundation, Inc.
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

/*******************************************************************************
 * This is a simple serial version of computing pi using the Monte Carlo Method
 * The model asks the user to enter the number of iterations to use for 
 * estimating pi
 * 
 * How to Setup Runtime Environment:
 *  $ qsub -I -l nodes=2:ppn=32:xe,walltime=01:30:00 -l advres=bwintern
 *
 * How to Compile:
 *  $ cc mpi_pi_MonteCarlo.c -o mpi_pi_MonteCarlo 
 *
 * How to Run:
 *  $ aprun -n 8 ./mpi_pi_MonteCarlo
 *
 * OutPut: make sure you run the program on multiple codes (-n ##). Also 
 * evrytime it ask you for the number of itteration. start with 1000, and add
 * a zero every other time you run the program.
 *
 ******************************************************************************/
// The following C libraries are needed.
#include <stdlib.h>  	// Is needed for exiting early if an error occurs
#include <stdio.h>   	// Is needed for printing the final results
#include <math.h>		// Is needed for fabs()/absolute value of floating point numbers
#include <string.h>  	// Is needed to 
#include <mpi.h>     	// Is needed for MPI parallelization
#define SEED 35791246	// SEED value for random number generator	

int main (int argc, char **argv){
    double start_time, end_time, time_diff;
    int num_iteration = 0;
    double x, y; 		// x,y value for the random coordinate
    int i;				// for loop counter
    int count; 			// Number of points in the first quadrant of unit circle
    double z;
	double pi=0.0; 		// holds approx value of pi
	double pi_tot; 		// holds final approx value of pi

	// MPI Initialization part
    int pRank;  // Every process will have a unique integer id/rank in the pool
    int pNum;   // How many processes are in the pool
    
	// MPI_Init(...) starts the parallel regions, here on the part of the code until
    // MPI_Finalize(...) call will be executed by all processes in the pool
    MPI_Init(&argc, &argv);
    
	// Next line will setup the number of processes in the communicator/pool. besed on -n ##
    MPI_Comm_size(MPI_COMM_WORLD, &pNum);
    
	// Next line will give a unique id to each process in the communicator/pool of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &pRank);
	
	if (pRank == 0) { 	// if I am process 0, then ask the user for number of itterations
    	printf("Please enter the number of iterations used to compute pi: \n");
    	scanf("%d",&num_iteration);
    }

	// Sending the number of iterations to all process in the communicator
	MPI_Bcast(&num_iteration, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Record the start time from here on:
    start_time = MPI_Wtime();

    srandom(time(0));		// Give srandom() a seed value
    count = 0;
    for(i=pRank+1; i<=num_iteration; i+=pNum){
        x = (double)random()/RAND_MAX; 		// Get a random x coordinate
        y = (double)random()/RAND_MAX; 		// Get a random y coordinate
        z = (x*x) + (y*y); 			   		// Check to see if number is inside unit circle
        if (z <=1.0) count++; 		   		// If it is, consider it a valid random point
        
        //num_iteration++;
    }
	
	// After each process is done with calculating their partial sum. the process 0 will ask all
	// other proceeses in the pool for their partial sums, and add them together to make a total
	// sum/value of pi.
	MPI_Reduce(&count, &pi_tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    //End recording the time here.
    end_time = MPI_Wtime();
    time_diff = end_time - start_time;
	
	if (pRank == 0){ // If I am process 0, then print  out the messages
    	
   		pi_tot=((double)count/(double)num_iteration*4);  // p = 4(m/n)
		printf("Number of iterations is = %d \n\n", num_iteration);
    	printf("Estimated value of PI is %g - (%17.15f)\n\n",pi_tot,pi_tot);
    	printf("Accurate value of PI from math.h is: %17.15f \n\n", M_PI);

    	printf("Difference between computed pi and math.h M_PI = %17.15f\n\n",
            fabs(pi_tot - M_PI));

    	printf("Time to compute = %f seconds\n\n", time_diff);
	}
	// Line is for end of parallel regions of the code.
    MPI_Finalize();

    return EXIT_SUCCESS;
}