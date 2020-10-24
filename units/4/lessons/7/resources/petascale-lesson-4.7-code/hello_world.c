/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 4: OpenMP
 * Lesson 7: OpenMP Applications & Practice
 * File: hello_world.c
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

