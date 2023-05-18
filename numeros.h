#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "linalg.h"
#include "images.h"

#define BATCH_SIZE 10000
#define TEST_SIZE 10000
#define ITERATIONS 500

#define strequ !strcmp

void train(void);
void test(void);
void image(char *path);
double mark(Matrix *output, unsigned char *answers);
