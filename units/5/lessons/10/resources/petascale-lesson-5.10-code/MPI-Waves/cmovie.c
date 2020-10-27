/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 5: MPI
 * Lesson 10: Wave Propagation in MPI
 * File: MPI-Waves/cmovie.c
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
    int frame_length;
    int nfchar;

    int nframe;

    character filename[26];

    frame_length = 256*256;

    float wave[256*256];

    FILE  *fmovie_ptr;
    FILE  *fheader_ptr;

    fheader_ptr = fopen("data/pebble.hdr","rb");

    if(fheader_ptr == NULL) 
       {
            printf("Error! opening pebble.hdr file");
            exit(1);
       }

    fmovie_ptr = fopen("data/pebble.mvlp","rb");

    if(fmovie_ptr == NULL) 
       {
            printf("Error! opening pebble.mvlp file");
            exit(1);
       }

    fmovie_filename_ptr = fopen("data/pebble.filenames","rb");

    if(fmovie_filename_ptr == NULL) 
       {
            printf("Error! opening pebble.filenames file");
            exit(1);

/* 

    Read header file for movie cube
    We need movie frame size --  Nx by Nz

    We need the number of frames...  -- Nframes
*/
          fread(&nframe,sizeof(int),1, fheader_ptr);
          fread(&nx,sizeof(int),1, fheader_ptr);
          fread(&nz,sizeof(int),1, fheader_ptr);
          fread(&nfchar,sizeof(int),1, fheader_ptr);

          printf("  Movie loop has %d frames \n"  ,nframe);
          printf("  Movie frame has %d rows \n"   ,nz);
          printf("  Movie loop has %d columns \n" ,nx);
          printf("  Movie file name has %d characters \n" ,nfchar);
          printf("  Frame length is %d floats \n" ,frame_length);
    
          fclose(fheader_ptr);

/*   
    Now read all the movie frames  

*/
    frame_length = 256*256;

         for(int ifr=0;ifr<nframe;ifr++) 
         {

//   Now read actual movie file name 
          fgets(filename,25, fmovie_filename_ptr);

//   Now read actual movie loop frame
          fread(&wave,sizeof(float),frame_length, fmovie_ptr);

//   Find max floating point in array wave().
            float wave_maxp = 0.0;
            float wave_maxn = 0.0;
            for(int ixz=0;ixz<frame_length;ixz++) 
            {
              if( wave[ixz] >= wave_maxp ) wave_maxp = wave[ixz];
              if( wave[ixz] <= wave_maxn ) wave_maxn = wave[ixz];
            }

            printf(&filename);
            printf("  Frame  %d  " ,ifr);
            printf("  Wave Max Positive=  %f  " ,wave_maxp);
            printf("  Max Negative=  %f  \n" ,wave_maxn);
            
         }


          fclose(fmovie_ptr);

          return 0;

} /* end of main program */
