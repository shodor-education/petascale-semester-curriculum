/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 4: OpenMP
 * Lesson 5: Convolution in OpenMP (Heat Transfer example)
 * File: heatTransfer.c
 * Developed by Maria Pantoja for the Shodor Education Foundation, Inc.
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
 
/*********************************************************0
Simplified implementation of a heat transfer problem using stencil
TempNew=Told+(Tup+Tbotton+Tright+Tleft)/SPEED
***********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
//#include <omp.h>
#include <assert.h>
#include <sys/mman.h>

#define REAL float
//Size of the Square Map, min size should be 100
//is a square matrix the size of the matrix will be NxN
#define N (4096)
//speed for heat transfer, a bigger number implies a slower speed of transfer
#define SPEED (10000) 
//number of times to run the stencil
#define TIMES (6553)

//coefficiennst for the stencil
const float ctop=0.5;
const float cbotton=0.5;
const float ceast=0.5;
const float cwest=0.5;

void init(REAL *buff) {
  int i, j;
  for (i= 0; i < N; i++) {
    for (j = 0; j < N; j++) {
        buff[i*N+j] = 0;
    }
  }
  //place three initial heat points on the array 
  //these heat points are to remain constant
  //buff[10][10] temp=100
  //buff[30][90] temp =200
  //buff[100][100] temp =150
  buff[9*N+9] = 100;
  buff[30*N+90] = 200;
  buff[100*N+100] = 150;
}


void heat_tranfer(REAL *restrict f1, REAL *restrict f2, int count) {
  REAL *f1_t = f1;
  REAL *f2_t = f2;

  for (int iter = 0; iter < count; iter++) {
    for (int y = 0; y < N; y++) {
      for (int x = 0; x < N; x++) {
        int center, top, bottom, east, west;
        center =  y*N+x;
        west = (x == 0)    ? center : center - 1;
        east = (x == N-1) ? center : center + 1;
        top = (y == 0) ? center : center - N;
        bottom=(y==N-1) ? center:center +N;

        f2_t[center] = f1_t[center] + (cwest * f1_t[west] + ceast * f1_t[east]
          + ctop * f1_t[top] + cbotton * f1_t[bottom])/SPEED;
      }
    }
    REAL *t = f1_t;
    f1_t = f2_t;
    f2_t = t;
    //original heat focus temperature remain constant
    f1_t[9*N+9] = 100;
    f1_t[30*N+90] = 200;
    f1_t[100*N+100] = 150;
  }
  return;
}

void print_result(REAL *f, char *out_path) {
  //FILE *out = fopen(out_path, "w");
  //assert(out);
  //size_t nitems = N*N;
  //fwrite(f, sizeof(REAL), nitems, out);
  //fclose(out);

  //only print first 10
  for (int i= 0; i < 10; i++) {
    printf("\n");
    for (int j = 0; j < 10; j++) {
        printf("%lf ",f[i*N+j]);
    }
  }
  printf("\n");

}

int main(int argc, char *argv[]) 
{
  
  struct timeval time_begin, time_end;


  REAL *f1 = (REAL *)malloc(sizeof(REAL)*N*N);
  REAL *f2 = (REAL *)malloc(sizeof(REAL)*N*N);
  assert(f1 != MAP_FAILED);
  assert(f2 != MAP_FAILED);
  //REAL *answer = (REAL *)malloc(sizeof(REAL)*N*N);
  REAL *f_final = NULL;

  f_final = (TIMES % 2)? f2 : f1;

  init(f1);
  print_result(f1, "heat.dat");
  printf("Running Heat Transfer Simulation %d times\n", TIMES); fflush(stdout);
  gettimeofday(&time_begin, NULL);
  heat_tranfer(f1, f2, TIMES);
  gettimeofday(&time_end, NULL);
  print_result(f_final, "heat.dat");

  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  

  fprintf(stderr, "Elapsed time : %.3f (s)\n", elapsed_time);
  
  free(f1);
  free(f2);
  return 0;
}
