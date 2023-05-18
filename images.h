#ifndef IMAGES_H
#define IMAGES_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <stdint.h>

// read_image
// ==========
//
// Reads a 24 bits-per-pixel bitmap image and returns the red values.
//
// Parameters:
//   path - Path to the image.
//
// Return:
//   Bytes corresponding to greyscale pixel values (1 byte per pixel),
//   left to right, then down. Call free() when this is no longer needed.
unsigned char* read_image(char* path);

#endif // IMAGES_H