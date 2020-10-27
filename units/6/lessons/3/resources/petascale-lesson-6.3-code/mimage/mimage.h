/* Blue Waters Petascale Semester Curriculum v1.0
 * File: mimage/mimage.h
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

#ifndef MIMAGE_H
#define MIMAGE_H 1

#include "misc.h"

//!  \brief mcolor is the basic color/pixel type
//!
typedef unsigned int mcolor;

//!  \brief mimage is the memory representation of an image
//!
typedef struct mimage_ mimage;

struct mimage_
{
  int nx, ny;
  int stride;
  mcolor *data;
};

//! \brief Create a new empty image of given size.
mimage *mimage_new(int nx, int ny);

//! \brief Free a mimage and its buffer.
void mimage_free(mimage *image);

//! \brief mimage_get_pixel -- get a pixel from an image.
//!
//! Note that this is inlined, so it will compile to direct code.
static inline mcolor mimage_get_pixel(mimage *image, int ix, int iy)
{
  return image->data[ix + image->stride * iy];
}

//! \brief mimage_set_pixel -- set a pixel in an image.
//!
//! Note that this is inlined, so it will compile to direct code.
static inline void mimage_set_pixel(mimage *image, int ix, int iy, mcolor pixel)
{
  image->data[ix + image->stride * iy] = pixel;
}

// File IO

//! \brief mimage_read_jpeg -- read image from JPEG file.
//!
mimage *mimage_read_jpeg(const char *filename);

//! \brief mimage_write_jpeg -- write image to JPEG file.
//!
int mimage_write_jpeg(mimage *image, const char *filename);

//! \brief mimage_read_png -- read image from PNG file.
//!
mimage *mimage_read_png(const char *filename);

//! \brief mimage_write_png -- write image to PNG file.
//!
int mimage_write_png(mimage *image, const char *filename);

// Color functions

//! \brief mcolor_init -- initialize a color
//!
//! Note that this is inlined, so it will compile to direct code.
static inline mcolor mcolor_init(int r, int g, int b, int a)
{
  return ((r&0xff)) | ((g&0xff)<<8) | ((b&0xff)<<16) | ((a&0xff)<<24) ;
}

//! \brief mcolor_get_red -- get red channel.
//!
//! Note that this is inlined, so it will compile to direct code.
static inline int mcolor_get_red(mcolor c)
{
  return c & 0xff;
}

//! \brief mcolor_get_green -- get green channel.
//!
//! Note that this is inlined, so it will compile to direct code.
static inline int mcolor_get_green(mcolor c)
{
  return (c>>8) & 0xff;
}

//! \brief mcolor_get_blue -- get blue channel.
//!
//! Note that this is inlined, so it will compile to direct code.
static inline int mcolor_get_blue(mcolor c)
{
  return (c>>16) & 0xff;
}

//! \brief mcolor_get_alpha -- get alpha channel.
//!
//! Note that this is inlined, so it will compile to direct code.
static inline int mcolor_get_alpha(mcolor c)
{
  return (c>>24) & 0xff;
}

//! Some standard colors
#define MCOLOR_BLACK   0xff000000
#define MCOLOR_WHITE   0xffffffff
#define MCOLOR_RED     0xff0000ff
#define MCOLOR_GREEN   0xff00ff00
#define MCOLOR_BLUE    0xffff0000
#define MCOLOR_YELLOW  0xff00ffff
#define MCOLOR_MAGENTA 0xffff00ff
#define MCOLOR_CYAN    0xffffff00


#endif
