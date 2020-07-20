#include <stdio.h> 
#include <omp.h>  /* Including OpenMp Library routines */
    int main(){
      omp_set_num_threads(5);  /* 5 threads specified by OpenMP call function */
      int numThreads, tNum;
     /*  Start of Parallel Section. Set tNum as private variable to avoid race condition.  */ 
      #pragma omp parallel private(tNum)        
      {
            tNum = omp_get_thread_num(); /* function call to get the individual thread numbers */
            if(tNum == 0) {
               numThreads = omp_get_num_threads(); /* function call to get the total number of threads */
               printf("Hello World! I am thread %d. There are %d threads.\n", tNum,numThreads);
            }
               else {
               printf("Hello World from thread %d.\n", tNum);
            }
      } 
      /*  End of Parallel Section */
return (0);
}


