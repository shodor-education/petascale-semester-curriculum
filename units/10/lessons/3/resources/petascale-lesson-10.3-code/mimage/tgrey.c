/* Blue Waters Petascale Semester Curriculum v1.0
 * File: mimage/tgrey.c
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

#include "mimage.h"

// See https://en.wikipedia.org/wiki/Mandelbrot_set
// This uses a naive algorithm. For better algorithms, see also
// https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set

#define INT(x) ((int)(x))
#define IMAGE_NX 512
#define IMAGE_NY 512
#define GMAX 512
/*
#define nx 512
#define ny 512
*/

// xorig,yorig -- center of image
// scale -- scale factor from x,y to pixels
static double xorig,yorig,scale;

static mcolor mandelbrot_color(double x, double y);
static double mapx(int ix);
static double mapy(int iy);
static void mandelbrot_init_color_map();

//
//   main program for image making 
//
int main(int argc, char **argv)
{

FILE  *frgbptr;
FILE  *fnameptr;
FILE  *fhdrptr;

// int   image[512][512];
//
//   function defined in the mimage.h file
//

//! \brief mimage_set_pixel -- set a pixel in an image.
//!
//! Note that this is inlined, so it will compile to direct code.
//static inline void mimage_set_pixel(mimage *image, int ix, int iy, mcolor pixel)
//{
//  image->data[ix + image->stride * iy] = pixel;
//}

// build a white screen background
//
//    open a file with an image
//
      

  fnameptr  =   fopen ("data/pebble.filenames","r");

  fhdrptr  =   fopen ("data/pebble.hdr","r");
int  nxf;
int  nzf;
int  nframef;
  fread(&nframef,sizeof(int),1,fhdrptr);
  fread(&nxf,sizeof(int),1,fhdrptr);
  fread(&nzf,sizeof(int),1,fhdrptr);
  fclose(fhdrptr);

  char filename[26];


  int wave_pixel;

// loop over movie frames

  for(int iframe=0;iframe<nframef;iframe++)
{
//  open frame file
  fgets(filename,26,fnameptr);
  frgbptr      =   fopen (filename,"rb");
 mimage *image = mimage_new(IMAGE_NX,IMAGE_NY);
 for(int iy=0;iy<image->ny;iy++)
   for(int ix=0;ix<image->nx;ix++)
     {
//      read the pixel value for this i,j image location

        fread(&wave_pixel,sizeof(int),1 ,frgbptr);

//      set the pixel value in the png image

        mimage_set_pixel(image, ix, iy, wave_pixel );
     }
//      set the file name
// mimage_write_png(image,"data/tgrey00001.png");
 filename[22] = 'p';
 filename[23] = 'n';
 filename[24] = 'g';
 mimage_write_png(image,filename);
 mimage_free(image);
 fclose(frgbptr);

} // end of movie loop conversion to png files

 fclose(fnameptr);

//
//   End of Main Program for Image Making.....
//
}




