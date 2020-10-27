/* Blue Waters Petascale Semester Curriculum v1.0
 * File: mimage/tpebble_junk.c
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

// xorig,yorig -- center of image
// scale -- scale factor from x,y to pixels
static double xorig,yorig,scale;

static mcolor mandelbrot_color(double x, double y);
static double mapx(int ix);
static double mapy(int iy);
static void mandelbrot_init_color_map();

int main(int argc, char **argv)
{
 xorig = 0.;

 yorig = 0.;
 scale = 3./IMAGE_NX;

 if(argc >= 3)
   {xorig = atof(argv[1]);
    yorig = atof(argv[2]);
   }
 if(argc == 4)
   {
    scale = scale/atof(argv[3]);
   }

 mandelbrot_init_color_map();
 mimage *image = mimage_new(IMAGE_NX,IMAGE_NY);
 for(int iy=0;iy<image->ny;iy++)
   for(int ix=0;ix<image->nx;ix++)
     {
      mimage_set_pixel(image, ix, iy, mandelbrot_color(mapx(ix),mapy(iy)));
     }
 for(int iy=(image->ny)/2;iy<(image->ny)/2+60;iy++)
   for(int ix=0;ix<iy+200;ix++)
/*
   for(int ix=0;ix<image->nx;ix++)
*/
     {
      mimage_set_pixel(image, ix,  iy, mandelbrot_color(mapx(ix),mapx(ix)));
     }
/*
      mimage_set_pixel(image, ix, 500, mandelbrot_color(mapx(ix),mapx(ix)));
      mimage_set_pixel(image, ix, 501, mandelbrot_color(mapx(ix),mapx(ix)));
      mimage_set_pixel(image, ix, 502, mandelbrot_color(mapx(ix),mapx(ix)));
      mimage_set_pixel(image, ix, 503, mandelbrot_color(mapx(ix),mapx(ix)));
      mimage_set_pixel(image, ix, 504, mandelbrot_color(mapx(ix),mapx(ix)));
      mimage_set_pixel(image, ix, 505, mandelbrot_color(mapx(ix),mapx(ix)));
*/
 mimage_write_png(image,"tpebble.png");
 mimage_free(image);
}

static double mapx(int ix)
{
 return xorig + scale*(ix-IMAGE_NX/2);
}

static double mapy(int iy)
{
 return yorig + scale*(iy-IMAGE_NY/2);
}

#define KMAX 6
static int mandelbrot(double x0, double y0)
{
 double x,y;
 x = 0.;
 y = 0.;

 int k = 0;
 while( ((x*x+y*y) < 4.) && (k < KMAX*5) )
   {
    double x1 = x*x-y*y + x0;
    double y1 = 2.*x*y + y0;
    x = x1;
    y = y1;
    k = k + 1;
   }
 if(k >= KMAX*5) return KMAX;

 return k % KMAX;
}

static mcolor mandelbrot_color_map[KMAX+1];

static mcolor mandelbrot_color(double x, double y)
{
 int mandelbrot_count = mandelbrot(x, y);
// printf("x %lf y %lf count %d\n",x,y,mandelbrot_count);
 return mandelbrot_color_map[mandelbrot_count];
}

static void mandelbrot_init_color_map()
{
 mandelbrot_color_map[5] = MCOLOR_RED;
 mandelbrot_color_map[4] = MCOLOR_YELLOW;
 mandelbrot_color_map[3] = MCOLOR_GREEN;
 mandelbrot_color_map[2] = MCOLOR_CYAN;
 mandelbrot_color_map[1] = MCOLOR_MAGENTA;
 mandelbrot_color_map[0] = MCOLOR_BLUE;
 mandelbrot_color_map[6] = MCOLOR_BLACK;

}

