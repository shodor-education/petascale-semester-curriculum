/*
 * Program to generate a .ppm file with an image of the Mandelbrot set
 *
 * compile with: gcc -Wall -o mandelbrot mandelbrot.c
 * run with: ./mandelbrot output.ppm
 *    where output.ppm is the desired name for the output file
 */

#include <stdio.h>
#include <stdlib.h>

int** pixels;                   //2D array of pixels
int numRows = 800;              //number of rows in image
int numCols = 800;              //number of columns in image

int mandelbrot(double x, double y) {
  //compute the color value for (x,y) in the black and white Mandelbrot set

  int maxIteration = 1000;
  int iteration = 0;

  double re = 0;   //current real part
  double im = 0;   //current imaginary part
  while((re*re + im*im <= 4) && (iteration < maxIteration)) {
    double temp = re*re - im*im + x;
    im = 2*re*im + y;
    re = temp;

    iteration++;
  }

  if(iteration != maxIteration)
    return 255;                       //corresponds to white pixel
  else
    return 0;                         //corresponds to black pixel
}

int main(int argc, char** argv) {
  if(argc != 2) {  //check number of arguments
    fprintf(stderr, "Usage: %s filename\n", argv[0]);
    exit(1);
  }

  //open the output file
  FILE* outfile; 
  outfile = fopen(argv[1], "wb");
  if(!outfile) {
    fprintf(stderr, "Unable to open output file: %s\n", argv[1]);
    exit(1);
  }

  //create array of pixels to store image
  pixels = malloc(numCols*sizeof(int*));
  for(int i=0; i < numCols; i++)
    pixels[i] = malloc(numRows*sizeof(int));

  double x, y;  //real-valued coordinates of the point

  //set pixels
  for (int j = 0; j < numRows; j++) {
    for (int i = 0; i < numCols; i++) {
      x = ((double)i / numCols -0.5) * 2;
      y = ((double)j / numRows -0.5) * 2;

      pixels[i][j] = mandelbrot(x,y);
    }
  }

  //print the image to the output file; first the header
  fprintf(outfile, "P2\n");    //magic number that denotes .ppm file
  fprintf(outfile, "%d %d\n", numCols, numRows);  //width and height
  fprintf(outfile, "255\n");                      //maximum color value

  //now print the values in row-major order
  for (int j = 0; j < numRows; j++) { 
    //  fprintf(outfile,"\n");
    for (int i = 0; i < numCols ; i++) {  
      fprintf(outfile, "%d ", pixels[i][j]); 
    } 
    fprintf(outfile, "\n"); 
  } 
  fclose(outfile); 
}
