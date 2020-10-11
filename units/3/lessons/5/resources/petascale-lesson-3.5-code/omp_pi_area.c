/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 3: Using a Cluster
 * Lesson 5: Running Code on a Cluster 1
 * omp_pi_area.c
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

/**************************************************************
 * Filename: omp_pi_area.c
 * Description: first intro to OpenMP directives
 * How to compile:
 *  $ cc omp_pi_area.c -o omp_pi_area.exe
 *
 * How to set number of threads to 8:
 *  $ export OMP_NUM_THREADS=8
 * How to Run on Blue Waters: 
 *  $ aprun -n 1 -d 8 ./omp_pi_area.exe
 *
 * How to Run on Cedar Supercomputer:
 *  $ export OMP_NUM_THREADS=8
 *  $ ./omp_pi_area.exe
 *************************************************************/
/*******************************************************************************
 * This is a simple program that shows the numerical integration example.
 * It computes pi by approximating the area under the curve:
 *      f(x) = 4 / (1+x*x) between 0 and 1.
 * To do this intergration numerically, the interval from 0 to 1 is divided into
 * some number (num_rect) subintervals and added up the area of
 * rectangles
 * The larger the value of the num_rect the more accurate your result
 * will be.
 *
 * The program first asks the user to input a value for subintervals, it
 * computes the approximation for pi, and then compares it to a more 
 * accurate aproximate value of pi in the math.h library.
 ******************************************************************************/
// The following C libraries are needed.
#include <stdio.h>	 // Is needed for printing the final results
#include <stdlib.h>  // Is needed for exiting early if an error occurs
#include <math.h>   // Is needed for fabs()/absolute value of floating point numbers
#include <omp.h>

int main(int argc, char *argv[]) {
    int num_rect = 10;   // number of rectangles
    double x_midp, pi;
    double sum = 0.0;
    double rect_width;
	int i;
    double start_t, end_t, compute_t;

    printf("Please enter the number of rectangles to compute pi: \n");
    scanf("%d",&num_rect);

    rect_width = 1.0/(double)num_rect;
    
    start_t = omp_get_wtime();
#pragma omp parallel shared(num_rect, rect_width) private(x_midp,i) reduction(+:sum)
    {
    #pragma omp for schedule(dynamic, 4000)
        for(i=0; i < num_rect; i++){
            //#pragma omp critical
            x_midp = (i+0.5)*rect_width;
            sum += 4.0/(1.0+x_midp*x_midp);
        }
    } // END: pragma
    pi = rect_width * sum;
    
    end_t = omp_get_wtime();
    compute_t = end_t - start_t;

    // print the result here: 
    printf("Results are: \n");
    printf("\t Computed pi is: %g (%17.15f)\n\n", pi,pi);
    printf("\t M_PI value from math.h is: %17.15f \n\n", M_PI);
    printf("\t Difference between pi and M_PI = %17.15f\n\n",
            fabs(pi - M_PI));
    printf("\t Total time to compute the for loop was: %lf \n", compute_t);
    return EXIT_SUCCESS;
}

