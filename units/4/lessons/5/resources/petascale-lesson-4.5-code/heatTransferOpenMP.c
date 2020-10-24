/*********************************************************0
Simplified implementation of a heat transfer problem using stencil
TempNew=Told+(Tup+Tbotton+Tright+Tleft)/SPEED
***********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
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
//number of threads 
#define THREADS (16)

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
  //place three initial heat points on the arrasimdy 
  //these heat points are to remain constant
  //buff[10][10] temp=100
  //buff[30][90] temp =200
  //buff[100][100] temp =150
  buff[9*N+9] = 100;
  buff[30*N+90] = 200;
  buff[100*N+100] = 150;
}


void heat_tranfer(REAL *restrict f1, REAL *restrict f2, int count) {
    #pragma omp parallel
    {
    REAL *f1_t = f1;  
    REAL *f2_t = f2;
    for (int iter = 0; iter < count; iter++) { 
	#pragma omp for //collapse(2) schedule(dynamic,8)
	for (int y = 0; y < N; y++) {
          #pragma omp simd
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
   }
  return;
}

void print_result(REAL *f, char *out_path, char *filename) {
  FILE *out = fopen(out_path, "w");
  assert(out);
  size_t nitems = N*N;
  fwrite(f, sizeof(REAL), nitems, out);
  fclose(out);
  //only frint fist 10
  for (int i= 0; i < 10; i++) {
    printf("\n");
    for (int j = 0; j < 10; j++) {
        printf("%lf ",f[i*N+j]);
    }
  }
  printf("\n");
  //create an image of the result
  FILE* pgmimg; 
  pgmimg = fopen(filename, "wb");
  
  REAL temp;
  
  // Writing Magic Number to the File 
  fprintf(pgmimg, "P2\n");  
  
  // Writing Width and Height 
  fprintf(pgmimg, "%d %d\n", 100, 100);  
  
  // Writing the maximum gray value 
  fprintf(pgmimg, "255\n");  
  for (int i = 0; i < 100 ; i++) {  
        fprintf(pgmimg,"\n");
	//printf("\n");
	for (int j = 0; j < 100; j++) { 
		temp = f[i*N+j]; 
                //make sure temp value is below 255 to be able to display controlled by SPEED
            	// Writing the gray values in the 2D array to the file 
                //printf("%d ", (int)temp); 
            	fprintf(pgmimg, "%d ", (int)temp); 
        } 
        fprintf(pgmimg, "\n"); 
    } 
    fclose(pgmimg); 

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
  print_result(f1, "heatInit.dat","init.pgm");
  
  printf("Running Heat Transfer Simulation %d times\n", TIMES); fflush(stdout);
  gettimeofday(&time_begin, NULL);
  heat_tranfer(f1, f2, TIMES);
  gettimeofday(&time_end, NULL);
  print_result(f_final, "heatFinal.dat","final.pgm");

  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  

  fprintf(stderr, "Elapsed time : %.3f (s)\n", elapsed_time);
  
  free(f1);
  free(f2);
  return 0;
}
