/* Blue Waters Petascale Semester Curriculum v1.0
 * File: mimage/mpng.c
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

/** File: routines to read and write PNG files. */
#include "mimage.h"

#include "png.h"

static int debug_flag = 0;

int mimage_write_png_stream(mimage *src, FILE *f);

int mimage_write_png(mimage *src, const char *filename)
{
  FILE *f = fopen(filename, "w");
  if(f == NULL)
    {
      misc_warn("cant write image to file %s\n",filename);
      return 1;
    }

  int status = mimage_write_png_stream(src, f);

  fclose(f);
  return status;
}

int mimage_write_png_stream(mimage *src, FILE *f)
{
  png_structp png_ptr = png_create_write_struct
                        (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr)
    {
      misc_warn("cant create PNG write struct\n");
      return 1;
    }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    {
      png_destroy_write_struct(&png_ptr,
                               (png_infopp)NULL);
      misc_warn("cant create PNG info struct\n");
      return 1;
    }


  png_init_io(png_ptr, f);
  png_set_IHDR(png_ptr, info_ptr, src->nx, src->ny, 8, PNG_COLOR_TYPE_RGB_ALPHA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);

  int j;
  for(j=0; j<src->ny; j++)
    png_write_row(png_ptr, (png_byte *) (src->data + j*src->stride));

//    printf("png wrote %d %d\n",src->nx,src->ny);
//    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

  png_write_end(png_ptr, info_ptr);

  png_destroy_write_struct(&png_ptr, &info_ptr);
  return 0;
}
/*

   If this returns NULL then an error has occured.

   TO DO:

     Do not print an error message. Instead, set a
   flag that could be probed with a function such as
   mimage_read_png_status().

*/

#define SUPPORT_PNG_PALETTE
mimage *mimage_read_png_stream(FILE *f)
{
  png_structp png_ptr;
  png_infop info_ptr;
//   unsigned int sig_read = 0;
  png_uint_32 width, height;
  int bit_depth, color_type, interlace_type;

#define PNG_BYTES_TO_CHECK 4
  png_byte header[PNG_BYTES_TO_CHECK];

  if(fread(header, 1, PNG_BYTES_TO_CHECK, f) != PNG_BYTES_TO_CHECK)
    {
      misc_warn("PNG cannot read header.\n");
      return NULL;
    }
  int is_png = !png_sig_cmp(header, (png_size_t) 0, PNG_BYTES_TO_CHECK);
  if (!is_png)
    {
      misc_warn("Does not appear to be a PNG file\n");
      return NULL;
    }
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (png_ptr == NULL)
    {
      return NULL;
    }



  /* Allocate/initialize the memory for image information.  REQUIRED. */
  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL)
    {
      png_destroy_read_struct(&png_ptr, NULL, NULL);
      return NULL;
    }

  png_init_io(png_ptr, f);

  png_set_sig_bytes(png_ptr, PNG_BYTES_TO_CHECK);

  png_read_info(png_ptr, info_ptr);

  png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
               &interlace_type, NULL, NULL);

  png_set_strip_16(png_ptr);
  png_set_packing(png_ptr);


  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png_ptr);

  mimage *img = mimage_new(width, height);
  switch(color_type)
    {
    case PNG_COLOR_TYPE_GRAY:
    {
      if(debug_flag)
        printf("png color type gray\n");
      png_byte *buf = misc_alloc_array(png_byte, width);

      int j;
      for(j=0; j<img->ny; j++)
        {
          png_read_row(png_ptr, buf, NULL);
          int i;
          for(i=0; i<(int)width; i++)
            {
              unsigned int b = (int) buf[i];
              img->data[i+j*img->stride] = mcolor_init(b,b,b,255);
            }
        }
      misc_free(buf);
      break;
    }
#ifdef SUPPORT_PNG_PALETTE
    case PNG_COLOR_TYPE_PALETTE:
    {
      if(debug_flag)
        printf("png color type palette\n");
      png_byte *buf = misc_alloc_array(png_byte, width);

      {
        int numisc_palette;
        struct png_color_struct *palette;
        png_get_PLTE(png_ptr, info_ptr, &palette, &numisc_palette);

        int j;
        for(j=0; j<img->ny; j++)
          {
            png_read_row(png_ptr, buf, NULL);
            int i;
            for(i=0; i<(int) width; i++)
              {
                png_byte b = buf[i];
                if(b > numisc_palette)
                  b = 0;
                struct png_color_struct c = palette[b];
                img->data[i+j*img->stride] =
                  mcolor_init(c.red, c.green, c.blue, 255);
              }
          }
      }
      misc_free(buf);
      break;
    }
#endif
    case PNG_COLOR_TYPE_RGB:
    {
      if(debug_flag)
        printf("png color type rgb\n");
      png_byte *buf = misc_alloc_array(png_byte, width*3);

      int j;
      for(j=0; j<img->ny; j++)
        {
          png_read_row(png_ptr, buf, NULL);
          int i;
          for(i=0; i<img->nx; i++)
            {
              png_byte *p = buf + i*3;
              img->data[i+j*img->stride] = mcolor_init(p[0],p[1],p[2],255);
            }
        }
      misc_free(buf);
      break;
    }
    case PNG_COLOR_TYPE_RGB_ALPHA:
    {
      if(debug_flag)
        printf("png color type rgba\n");
      int j;
      for(j=0; j<img->ny; j++)
        {
          png_byte *row[1];
          row[0] = (png_byte *) (img->data + j*img->stride);
          png_read_rows(png_ptr, row, NULL, 1);
        }
      break;
    }

    default:
      misc_warn("Unknown PNG type %d\n",color_type);
    }

  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

  return img;
}

// TO DO: set error status if error occurs.
mimage *mimage_read_png(const char *filename)
{
  FILE *f = fopen(filename, "r");

  if (!f)
    {
      misc_warn("Cannot read PNG image %s\n",filename);
      return NULL;
    }

  mimage *img = mimage_read_png_stream(f);

  fclose(f);
  return img;
}
