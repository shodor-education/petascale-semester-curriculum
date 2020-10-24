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
#define SPEED (1000) 
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
  //buff[2]2] temp=10
  //buff[3][9] temp =100
  //buff[6][6] temp =50
  buff[2*N+2] = 10;
  buff[3*N+9] = 100;
  buff[6*N+6] = 50;
}


void heat_tranfer(REAL *restrict f1, REAL *restrict f2, int count) {
  {
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
      f1_t[2*N+2] = 10;
      f1_t[3*N+9] = 100;
      f1_t[6*N+6] = 50;
    }
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
