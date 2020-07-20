#include <stdio.h> 
#include <omp.h>  /* Including OpenMp Library routines */
int main ()
{
   int number_of_threads, this_thread, iteration;
   int omp_get_max_threads(), omp_get_thread_num();
   number_of_threads = omp_get_max_threads();
   fprintf(stderr, "%2d threads\n", number_of_threads);
# pragma omp parallel for default(private) shared(number_of_threads)
   for (iteration = 0; 
         iteration < number_of_threads; iteration++) {
         this_thread = omp_get_thread_num();
         fprintf(stderr, "Iteration %2d, thread %2d: Hello, world!\n",      
         iteration, this_thread);
   }
}

