/* Blue Waters Petascale Semester Curriculum v1.0
 * File: mimage/sprintest.c
 * Developed by R. Phillip Bording for the Shodor Education Foundation, Inc.
 * Included in the following lessons:
 * - Unit 5 (MPI) Lesson 10: Wave Propagation in MPI
 * - Unit 6 (Hybrid MPI + OpenMP) Lesson 3: Pebble in Pond Wave Equation
 * - Unit 8 (OpenACC) Lesson 1: Accelerating Scientific Applications
 * - Unit 10 (Productivity and Visualization) Lesson 3: Visualization 1
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
#include <string.h>
 
int main( ) 
{
   int value = 50 ; 
   float flt = 7.25 ; 
   char c = 'Z' ; 
   char cn[10] = {'\0'} ; 
   char string[40] = {'\0'} ; 
   char basefilename[50] = {'\0'} ; 
   char filename[25] = {'\0'} ; 
   

   printf ( "int value = %d \n char value = %c \n " \
            "float value = %f", value, c, flt ) ; 

/*   Now, all the above values are redirected to string 
     instead of stdout using sprint */
 
   printf("\n Before using sprint, data in string is %s", string);
   sprintf ( string, "%d %c %f", value, c, flt );
   printf("\n After using sprint, data in string is %s", string);
   printf("\n ");

   filename[0] = 'g';
   filename[1] = 'r';
   filename[2] = 'a';
   filename[3] = 'y';
   filename[4] = '_';

   basefilename[0] = 'd';
   basefilename[1] = 'a';
   basefilename[2] = 't';
   basefilename[3] = 'a';
   basefilename[4] = '/';

   for (int icm=0;icm<5;icm++ )
      {
        basefilename[icm+5] = filename[icm];
      }

   printf("\n Before using sprint, data in basefilename string is %s x", basefilename);
   sprintf ( basefilename, "%10c", 'zzzzz10005' );
   printf("\n After using sprint, data in basefilename string is %s x", basefilename);
   printf("\n Before using sprint, data in filename string is %s x", filename);

   printf("\n ");
   return 0;
}
