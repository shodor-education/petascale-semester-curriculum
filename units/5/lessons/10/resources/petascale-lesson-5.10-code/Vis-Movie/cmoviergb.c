/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 10: Productivity and Visualization
 * Lesson 3: Visualization 1
 * File: Vis-Movie/cmoviergb.c
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

      this version does a float to argb conversion
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

    int nframe;

//    frame_length = nx*nz;

    float wave[512*512];

    int   wave_rgb[512*512];

    char  filename[26];

    FILE  *fmovie_ptr;
    FILE  *fmovie_rgb_ptr;
    FILE  *fheader_ptr;
    FILE  *fmovie_filenames_ptr;

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

    fmovie_filenames_ptr = fopen("data/pebble.filenames","r");
    if(fmovie_filenames_ptr == NULL) 
       {
            printf("Error! opening pebble.filenames file");
            exit(1);
       }
/* 

    Read header file for movie cube
    We need movie frame size --  Nx by Nz

    We need the number of frames...  -- Nframes
*/
          fread(&nframe,sizeof(int),1, fheader_ptr);
          fread(&nx,sizeof(int),1, fheader_ptr);
          fread(&nz,sizeof(int),1, fheader_ptr);

          frame_length = nx*nz;

          printf("  Movie loop has %d frames \n"  ,nframe);
          printf("  Movie frame has %d rows \n"   ,nz);
          printf("  Movie loop has %d columns \n" ,nx);
          printf("  Frame length is %d floats \n" ,frame_length);
    
          fclose(fheader_ptr);

/*   
    Now read all the movie frames  
*/

//         printf("\n");
         for(int ifr=0;ifr<nframe;ifr++) 
     {

//   Now read actual movie loop frame

//   read in file name so we can create the argb output file

//   Find max floating point in array wave().

              fgets(filename,26, fmovie_filenames_ptr);
//             printf("File name ");
//             printf(&filename);
//             filename[22] = "p";
//             filename[23] = "n";
//             filename[24] = "g";
//             printf(&filename);

              float wave_maxp = 0.0;
              float wave_maxm = 0.0;

              fread(&wave,sizeof(float),frame_length , fmovie_ptr);

              for(int ixz=0;ixz<frame_length;ixz++) 
              {
                if( wave[ixz] > wave_maxp ) wave_maxp = wave[ixz];
                if( wave[ixz] < wave_maxm ) wave_maxm = wave[ixz];
              }
//            printf(" %6d Wave Max +Pos = %f " ,ifr,wave_maxp);
//            printf(" Max -Neg = %f \n" ,wave_maxm);

              fmovie_rgb_ptr = fopen(filename,"wb");

/*
   scale floating point into =>  -127 -- +128 -- 256 = 8 bits grey scale
*/
//    make the conversion to pixel values -- for png file
    float pmax;
    float smax;
    float smin;

    int   irgb;

    pmax  = fmaxf(wave_maxp,abs(wave_maxm));

    smax = 2.0*pmax;
    smin = 0.0;
/*
                         hex
                         a r g b
          grey scale  =  ff000000 = black
          grey scale  =  ffffffff = white

          binary         44444444   bits = 32 bits
*/


     int ibit;
     int irbit;
     int igbit;
     int ibbit;
     int mone;
     int mrgb;
     float amu;

     mone = -256*256*256;

//          0.0   wave          pmax
//          0.0               256.0
//          0.0        0.5    1.0
//          wave_maxn  0.0    wave_maxp
//          -2.4       0.0    +2.7
//          0.0        2.7    +5.4
//           0                  -1       
//       ( -2          +2 )  0/4 0.0
//       (  0          +2 ) +2/4 0.5
//       (  1          +2 ) +3/4 0.75
//       (  2          +2 ) +4/4 1.0

              for(int ixz=0;ixz<frame_length;ixz++) 
              {
//                      floating point
                amu =  (wave[ixz]+pmax)/smax;
                irbit   = 256.0*amu;
                igbit   = 256.0*amu;
                ibbit   = 256.0*amu;
                irbit   = 0;
                igbit   = 0;

                irgb = 256*256*irbit + 256*igbit + ibbit;
                mrgb = mone + irgb;

//              printf("  %f    %d \n  ",wave[ixz],mrgb);
//                      rgb pixels 
                wave_rgb[ixz] = mrgb;
              }

//              write rgb pixel file

              fwrite(&wave_rgb,sizeof(int),frame_length , fmovie_rgb_ptr);
            
              fclose(fmovie_rgb_ptr);

         }  //  End of Movie frame making.......



          fclose(fmovie_ptr);

          return 0;

} /* end of main program */



///*    cfrgb.c is a program to map 
//      floating point range 
//      into argb integer
//
//      test code to prepare for doing tpebble.c
//
//      Phil Bording
//      July 16, 2020
//      Blue Waters Project
//
//
//#include <stdlib.h>
//#include <string.h>
//#include <stdio.h>
//#include <math.h>
//
//
//int main()
//{
//
//    data maximum positive number
//    data maximum negative number
//
//    float dmaxp;
//    float dmaxm;
//
//    float pmax;
//
//    float smax;
//    float smin;
//
//
////    end result argb integer
//
//    int   irgb;
//
//    dmaxp =  2.6;
//    dmaxm = -2.2;
//
//    pmax  = fmaxf(dmaxp,abs(dmaxm));
//
//    smax = 2.0*pmax;
//    smin = 0.0;
///*
//                        hex
//.
//                         a r g b
//         grey scale  =  ff000000 = black
//          grey scale  =  ffffffff = white
//
//          binary         44444444   bits = 32 bits
//
//*/
//
////   float sgrey = 0.0;
//   float dgrey = 1.0;
//
//    float smov = -0.5*smax;
//    float dmov =  smax/256.0;
//
//     int ibit;
//     int mone;
//     int mrgb;
//
//     mone = -256*256*256;
//     for(int  i=1;i<=255;i=i+1) 
//      {
//         irgb = 256*256*i + 256*i + i;
//         mrgb = mone + irgb;
//         printf("  %-9.3f   %-9f  %-10d %-9x  \n",sgrey,smov,mrgb,mrgb);
//         sgrey = sgrey + dgrey;
//         smov  = smov +dmov;
//      }
//
//    return 0;
//
//} /* end of main program */
