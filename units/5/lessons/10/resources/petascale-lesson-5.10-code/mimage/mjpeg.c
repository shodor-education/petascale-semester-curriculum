/* Blue Waters Petascale Semester Curriculum v1.0
 * File: mimage/mjpeg.c
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

/** File: routines to read and write JPEG files. */
#include "mimage.h"

mimage *mimage_read_jpeg(const char *filename);
int mimage_write_jpeg(mimage *image, const char *filename);

#include "jpeglib.h"

typedef unsigned char uchar;
static boolean print_text_marker (j_decompress_ptr cinfo)
{
  printf("found a marker\n");
}

mimage *mimage_read_jpeg(const char *filename)
{
  mimage *image = NULL;
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  FILE *file;
  file = fopen(filename, "r");
  if(file == NULL) return NULL;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_set_marker_processor(&cinfo, JPEG_COM, print_text_marker);
  jpeg_set_marker_processor(&cinfo, JPEG_APP0+12, print_text_marker);
  jpeg_stdio_src(&cinfo, file);
  jpeg_read_header(&cinfo, TRUE);

// printf("image nx %d ny %d channels %d\n",
//     cinfo.image_width,cinfo.image_height,cinfo.num_components);
  image = mimage_new(cinfo.image_width,cinfo.image_height);

#define MAX_BUFS 4

  int rowlength = image->nx * 4;
  uchar *buffer = misc_alloc_array(uchar, rowlength*MAX_BUFS);

  JSAMPROW buf[MAX_BUFS];
  for(int i=0; i<MAX_BUFS; i++)
    buf[i] = (JSAMPROW) (buffer + i*rowlength);

  jpeg_start_decompress(&cinfo);

  int j = 0;
  while (cinfo.output_scanline < cinfo.output_height)
    {
      int num_scanlines = jpeg_read_scanlines(&cinfo, buf, MAX_BUFS);

      for(int is=0; is<num_scanlines; is++)
        {
          JSAMPLE *line = buf[is];
          JSAMPLE channels[10];
          int l = 0;
          for(int i=0; i<image->nx; i++)
            {
              for(int channel=0; channel<cinfo.num_components; channel++)
                channels[channel] = line[l+channel];
              l += cinfo.num_components;
              int r,g,b,a;
              r = channels[0];
              g = channels[1];
              b = channels[2];
              a = (cinfo.num_components >= 4) ? channels[3] : 255;

              mcolor color = mcolor_init(r,g,b,a);
              image->data[i + image->stride*j] = color;
            }
          j++;
        }

//    printf("scanlines %d row %d\n",num_scanlines, j);
    }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  misc_free(buffer);
  return  image;
}

int mimage_write_jpeg(mimage *image, const char *filename)
{
  int quality = 100;  // supported values are 0-100

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE * file;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  file = fopen(filename, "w");
  if(file == NULL) return 1;
  jpeg_stdio_dest(&cinfo, file);

  cinfo.image_width = image->nx;
  cinfo.image_height = image->ny;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);

  int row_stride = image->nx * 3;
  JSAMPLE *buffer = misc_alloc_array(unsigned char, row_stride * 4);
  JSAMPROW buf[4];
  buf[0] = buffer;
  buf[1] = buffer + row_stride;
  buf[2] = buffer + 2*row_stride;
  buf[3] = buffer + 3*row_stride;

  jpeg_set_quality(&cinfo, quality, TRUE);
  jpeg_start_compress(&cinfo, TRUE);
  int count = image->ny;
  int row = 0;
  while (cinfo.next_scanline < cinfo.image_height)
    {
      int jcount = 0;
      if(count >= 4)
        jcount = 4;
      else if(count >= 2)
        jcount = 2;
      else if(jcount > 0)
        jcount = 1;

      for(int jrow=0; jrow<jcount; jrow++)
        {
          JSAMPLE *line = buf[jrow];
          for(int i=0; i<image->nx; i++)
            {
              mcolor color = image->data[i + image->stride * row];
              int r = mcolor_get_red(color);
              int g = mcolor_get_green(color);
              int b = mcolor_get_blue(color);
              line[0] = r;
              line[1] = g;
              line[2] = b;
              line += 3;
            }
          row++;
        }

      jpeg_write_scanlines(&cinfo, buf, jcount);
    }
  misc_free(buffer);
  jpeg_finish_compress(&cinfo);
  fclose(file);
  return 0;
}



