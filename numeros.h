#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "linalg.h"
#include "images.h"

#define BATCH_SIZE 10000
#define TEST_SIZE 10000
#define ITERATIONS 500

#define strequ !strcmp

// train
// =====
//
// Trains the model using the MNIST database.
void train(void);

// test
// ====
//
// Uses the 'brainsave' file created by train() to test the model's accuracy.
void test(void);

// image
// =====
//
// Attempts to determine the number contained in a bitmap.
//
// Parameters:
//   path - The path to a 28x28, 24bpp greyscale image.
void image(char *path);

// mark
// ====
//
// Marks the output of either test() or train() against the actual answers.
//
// Parameters:
//    output - The matrix output of either test() or train().
//   answers - An array of answers, where each byte is the next image's number.
//
// Return:
//   A ratio between 0 and 1 representing correct answers over total images.
double mark(Matrix *output, unsigned char *answers);
