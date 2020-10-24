/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 4: OpenMP
 * Lesson 7: OpenMP Applications & Practice
 * File: helloWorld.c
 * Developed by Widodo Samyono for the Shodor Education Foundation, Inc.
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


