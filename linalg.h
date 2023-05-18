#ifndef LINALG_H
#define LINALG_H


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct
{
	unsigned int rows, cols;
	double *data;
} Matrix;

Matrix *matrix_new(unsigned int rows, unsigned int cols);
Matrix *matrix_new_from_data(unsigned int rows, unsigned int cols, double *data);
void matrix_free(Matrix *this);
void matrix_set(Matrix *this, unsigned int row, unsigned int col, double value);
double matrix_get(Matrix *this, unsigned int row, unsigned int col);
Matrix *matrix_multiply(Matrix *this, Matrix *other);
Matrix *matrix_elementwise_multiply(Matrix *this, Matrix *other);
Matrix *matrix_add_to_rows(Matrix *this, Matrix *other);
Matrix *matrix_sum_rows(Matrix *this);
Matrix *matrix_ReLU(Matrix *this);
Matrix *matrix_dReLU(Matrix *this);
Matrix *matrix_transpose(Matrix *this);
Matrix *matrix_subtract(Matrix *this, Matrix *other);
Matrix *matrix_multiply_scalar(Matrix *this, double value);
Matrix *matrix_softmax(Matrix *this);
void matrix_rand(Matrix *this);
void matrix_clear(Matrix *this);


#endif // LINALG_H
