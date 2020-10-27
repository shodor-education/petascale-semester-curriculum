/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 5: MPI
 * Lesson 10: Wave Propagation in MPI
 * File: MPI-Waves/cmoviehead.c
 * Developed by R. Phillip Bording for the Shodor Education Foundation, Inc.
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

/*    c program to read the pebble movie loop 
      floating point file

      test code to prepare for doing tpebble.c
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


int main()
{

    int nx;
    int nz;

    int nframe;
    int nfchar;


    FILE *fptr;

    fptr = fopen("data/pebble.hdr","rb");

    if(fptr == NULL) 
       {
            printf("Error! opening data/pebble.hdr file");
            exit(1);
       }

    nx = 0;
    nz = 0;
    nframe = 0;
/* 

    Read header file for movie cube
    We need movie frame size --  Nx by Nz

    We need the number of frames...  -- Nframes
*/
          fread(&nframe,sizeof(int),1, fptr);
          fread(&nx,sizeof(int),1, fptr);
          fread(&nz,sizeof(int),1, fptr);
          fread(&nfchar,sizeof(int),1, fptr);

           printf("  Movie loop has %d frames \n"  ,nframe);
           printf("  Movie frame has %d rows \n"   ,nz);
           printf("  Movie loop has %d columns \n" ,nx);
           printf("  Movie filename has %d characters \n" ,nfchar);

    
          fclose(fptr);

          return 0;

} /* end of main program */
