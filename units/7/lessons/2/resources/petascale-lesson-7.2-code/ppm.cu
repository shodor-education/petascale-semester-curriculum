/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 7: CUDA
 * Lesson 2: Image Processing
 * File: ppm.cu
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

#include "ppm.h"
#include <stdio.h>

// Helper function which loads an image
// This function returns a structure to a 
// ppm_t which holds the pixels which can be
// further manipulated.
ppm_t* loadPPMImage(const char* filepath){
    // Open a file
    FILE* fp = fopen(filepath,"r");
    // Check if the file was opened
    if(NULL==fp){
        printf("File not found! %s",filepath);
        exit(0);
    }
    
    // Create a PPM image
    ppm_t* image = (ppm_t*)malloc(sizeof(ppm_t));
    // Buffer to hold a line that we read in
    char line[80];
    // This variable below changes if we have read the
    // magic number (which is typically the first item
    // in the file), then we know we can start reading
    // in the width and height.
    int haveReadMagicNumber = 0;
    // Keep track of pixels read
    int pixelsRead =0;
    while(fgets(line,sizeof(line),fp)!=NULL){
        // Uncomment the line below for debugging
        // information if you want to see what is being read.
        // fprintf(stderr,"read: %s\n",line); 
        // Ignore any lines that start with a '#'
        if(line[0]=='#'){
            continue;
        }else if(line[0]=='P'){
            // Here we retrieve the character after
            // the 'P', and then we want the ascii
            // representation of that number
            image->magicnumber = line[1]-48;
            haveReadMagicNumber = 1;
        }else if(1==haveReadMagicNumber){
            haveReadMagicNumber=2;
            // Further separate the width and height
            char* pch;  // We need to split this string into
                        // two different 'tokens' or 'values'
            pch = strtok(line," ");
            image->width  = atoi(pch); 
            image->height = atoi(strtok(NULL, " "));
            // Once we have read the width and height
            // we can allocate memory for pixels.
            image->pixels = (unsigned char*)malloc(sizeof(unsigned char)*image->width * image->height*3);
        }else{
            // Now as we read pixel values in, we
            // store them in our array.
            // Note:    There is a little bit going in in this line
            //          in that we have to increment every pixel
            //          that we read, and we can do this in one
            //          line with the post-increment operator.
            image->pixels[pixelsRead++] = atoi(line);  
        } 
    }

    // Close our file when we are done with it.
    fclose(fp);
    // Finally return our pointer to our newly allocated
    // ppm_t which has all of our data.
    return image;
}

// Outputs a PPM image 
void savePPMImage(const char* filename, ppm_t* ppm){
    // Open a new file to write to
    // 'w+' will create a new file if it does not exist
    FILE* fp = fopen(filename,"w+"); 
    fprintf(fp,"P%d\n",ppm->magicnumber);
    fprintf(fp,"# Exported from your program!\n");
    fprintf(fp,"%d %d\n",ppm->width,ppm->height);
    // Now we simply write out every pixel to the file 
    int pixelsWritten =0;
    while(pixelsWritten < (ppm->width*ppm->height*3)){
        fprintf(fp,"%d\n",ppm->pixels[pixelsWritten++]);
    }
    fclose(fp);
}
