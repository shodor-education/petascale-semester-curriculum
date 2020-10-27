/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 5: MPI
 * Lesson 10: Wave Propagation in MPI
 * File: MPI-Waves/cfrgb.c
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

/*    c program to map 
      floating point range 
      into argb integer

      test code to prepare for doing tpebble.c
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


int main()
{

//    data maximum positive number
//    data maximum negative number

    float dmaxp;
    float dmaxm;

    float pmax;

    float smax;
    float smin;


//    end result argb integer

    int   irgb;

    dmaxp =  2.6;
    dmaxm = -2.2;

    pmax  = fmaxf(dmaxp,abs(dmaxm));

    smax = 2.0*pmax;
    smin = 0.0;
/*
                         hex
                         a r g b
          grey scale  =  ff000000 = black
          grey scale  =  ffffffff = white

          binary         44444444   bits = 32 bits

*/

    float sgrey = 0.0;
    float dgrey = 1.0;

    float smov = -0.5*smax;
    float dmov =  smax/256.0;

     int ibit;
     int mone;
     int mrgb;

     mone = -256*256*256;
     for(int  i=1;i<=255;i=i+1) 
      {
         irgb = 256*256*i + 256*i + i;
         mrgb = mone + irgb;
         printf("  %-9.3f   %-9f  %-10d %-9x  \n",sgrey,smov,mrgb,mrgb);
         sgrey = sgrey + dgrey;
         smov  = smov +dmov;
      }

    return 0;

} /* end of main program */
