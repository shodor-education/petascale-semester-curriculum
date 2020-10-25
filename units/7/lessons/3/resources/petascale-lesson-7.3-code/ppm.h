/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 7: CUDA
 * Lesson 3: Graphics Interop with OpenGL
 * File: ppm.h
 * Developed by Michael D. Shah for the Shodor Education Foundation, Inc.
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

#ifndef PPM_H
#define PPM_H


// The structure represents an image.
typedef struct{
    // The magic number tells us what type of PPM
    // image we are working with, that is, how is
    // the data formatted within the file.
    int magicnumber;
    // The width of our image
    int width;
    // The height of our image
    int height;
    // The range of values for a particular
    // component of a pixel.
    // For example, 0-255 is often used to tell
    // the intensity of the Red, green, and blue
    // color components of a pixel
    int range;
    // This array stores the raw values of the pixels
    unsigned char* pixels;
}ppm_t;

// Some helper functions found in ppm.cu
// We make these functions 'extern' because they are in an external module.
// Note that we need to compile the ppm.cu file
// which has the implementations of these functions.
// Anywhere we otherwise include 'ppm.h' we are able
// to use both of these functions.

extern ppm_t* loadPPMImage(const char* filepath);
extern void savePPMImage(const char* filename, ppm_t* ppm);

#endif
