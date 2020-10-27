/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 9: Optimization
 * Lesson 4: Multiprocessor Caching and False Sharing
 * File: ppm-balance.c
 * Developed by David P. Bunde for the Shodor Education Foundation, Inc.
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
 * Program to read a .ppm file and considers it split into regions,
 * each consisting of a block of rows, one per thread.  In parallel,
 * counts the number of black pizels (i.e. those with value 0) in each
 * region.  Nominally, this is to measure the load balance of the
 * program used to generate the input file, but the actual goal is to
 * demonstrate false sharing of the array in which the counts of black
 * pixels are stored.
 *
 * compile with: gcc -Wall -std=gnu99 -o ppm-balance ppm-balance.c -fopenmp
 * run with: ./ppm-balance input.ppm [num_threads [distance]]
 *    where
 *      input.ppm is the desired name for the output file
 *      num_threads is the number of threads to use (default=2)
 *      distance is how close the cells used for pixel counts are
 *               stored (default is 1, which makes them adjacent)
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

int** pixels;                   //2D array of pixels
int numRows;                    //number of rows in image
int numCols;                    //number of columns in image

void check(int correct, int actual, char* mesg) {
  if(correct != actual) {
    fprintf(stderr, "Input error: Error reading %s\n", mesg);
    exit(1);
  }
}
  
int main(int argc, char** argv) {
  int numThreads = 2;   //number of threads to use when counting black pixels
  int dist = 1;         //distance between cells used for pixel counts

  //read and validate command line arguments
  if(argc == 2) {
    //ok, no optional arguments
  } else if(argc == 3) {
    numThreads = atoi(argv[2]);
  } else if(argc == 4) {
    numThreads = atoi(argv[2]);
    dist = atoi(argv[3]);
  } else {
    fprintf(stderr, "Usage: %s filename [num_threads [distance]]\n", argv[0]);
    exit(1);
  }
  if(numThreads < 1) {
    fprintf(stderr, "Invalid number of threads: %d\n", numThreads);
    exit(1);
  }
  if(dist < 1) {
    fprintf(stderr, "Invalid distance between cells used for pixel counts: %d\n", dist);
    exit(1);
  }
  
  //open the output file
  FILE* infile;  //file from which input image is read 
  infile = fopen(argv[1], "rb");
  if(!infile) {
    fprintf(stderr, "Unable to open input file: %s\n", argv[1]);
    exit(1);
  }

  //read the image; first the header
  check(2, fscanf(infile, "P2%d %d\n", &numCols, &numRows), "width and height");
  int ignore;  //used to read the maximum color (not used in the program)
  check(1, fscanf(infile, "%d\n", &ignore), "maximum color");

  //create array of pixels to store image
  pixels = malloc(numCols*sizeof(int*));
  for(int i=0; i < numCols; i++)
    pixels[i] = malloc(numRows*sizeof(int));

  //read the pixels into the array
  for (int j = 0; j < numRows; j++) { 
    for (int i = 0; i < numCols ; i++) {  
      check(1, fscanf(infile, "%d ", &(pixels[i][j])), "a pixel"); 
    } 
  } 
  fclose(infile); 

  int numBlackSize = numThreads*dist;  //size of array used to store pixel counts
  int* numBlack = (int*) malloc(numBlackSize*sizeof(int));  //array itself
  for(int i=0; i < numBlackSize; i++)
    numBlack[i] = 0;

  struct timeval start;  //used to store start and stop times for the count
  struct timeval stop;

  gettimeofday(&start, NULL);  //get the start time

  //now do the counting in parallel
  #pragma omp parallel num_threads(numThreads)
  {
    int threadNum = omp_get_thread_num();       //number of this thread
    int actNumThreads = omp_get_num_threads();  //total number of threads
                                        //(hopefully, equal to numThreads)

    int low = numRows*threadNum/actNumThreads;     //range for row loop for this thread
    int high = numRows*(threadNum+1)/actNumThreads;

    for (int j = low; j < high; j++) {
      for (int i = 0; i < numCols; i++) {
	if(pixels[i][j] == 0)
	  numBlack[threadNum*dist]++;
      }
    }
  }

  gettimeofday(&stop, NULL);  //get the ending time
  long diff = (stop.tv_sec - start.tv_sec)*1000000 + (stop.tv_usec-start.tv_usec);
  printf("Time for counting: %ld microseconds\n", diff);

  //print the number of black pixels in each region
  for(int i=0; i < numBlackSize; i++)
    if(numBlack[i] != 0)
      printf("%d : %d\n", i, numBlack[i]);

  //free memory
  free(numBlack);
  for(int i=0; i < numCols; i++)
    free(pixels[i]);
  free(pixels);
}
