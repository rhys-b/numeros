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

// matrix_new
// ==========
//
// Allocates space and initializes a new 2D matrix.
//
// Parameters:
//   rows - The number of rows the matrix has.
//   cols - The number of columns the matrix has.
//
// Return:
//   The matrix. Call matrix_free() when no longer needed.
Matrix *matrix_new(unsigned int rows, unsigned int cols);

// matrix_new_from_data
// ====================
//
// Creates a 2D matrix from preexising data.
//
// Parameters:
//   rows - The number of rows the matrix has.
//   cols - The number of columns the matrix has.
//   data - The left to right then down data of the matrix. The matrix takes ownership of this data.
//
// Return:
//   The matrix. Call matrix_free() when no longer needed.
Matrix *matrix_new_from_data(unsigned int rows, unsigned int cols, double *data);

// matrix_free
// ===========
//
// Releases the resources used by a matrix.
//
// Parameters:
//   this - The matrix.
void matrix_free(Matrix *this);

// matrix_set
// ==========
//
// Sets a value in a matrix.
//
// Parameters:
//    this - The matrix.
//     row - The row.
//     col - The column.
//   value - The value to set the element to.
void matrix_set(Matrix *this, unsigned int row, unsigned int col, double value);

// matrix_get
// ==========
//
// Gets a value from a matrix.
//
// Parameters:
//   this - The matrix.
//    row - The row.
//    col - The column.
//
// Return:
//   The value in the matrix.
double matrix_get(Matrix *this, unsigned int row, unsigned int col);

// matrix_multiply
// ===============
//
// Multiplies this matrix with another matrix.
//
// Parameters:
//    this - The first matrix.
//   other - The second matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_multiply(Matrix *this, Matrix *other);

// matrix_elementwise_multiply
// ===========================
//
// Multiplies each element of a matrix with the corresponding element of another matrix.
//
// Parameters:
//    this - The first matrix.
//   other - The second matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_elementwise_multiply(Matrix *this, Matrix *other);

// matrix_add_to_rows
// ==================
//
// Adds the value in the first column of the other matrix to each element in the corresponding row of the first matrix.
//
// Parameters:
//    this - The first matrix.
//   other - The second matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_add_to_rows(Matrix *this, Matrix *other);

// matrix_sum_rows
// ===============
//
// Creates an (N,1) matrix where each element is the sum of the row of the input matrix.
//
// Parameters:
//   this - The matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_sum_rows(Matrix *this);

// matrix_ReLU
// ===========
//
// Performs a ReLU on each element of the matrix.
//
// Parameters:
//   this - The matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_ReLU(Matrix *this);

// matrix_dReLU
// ============
//
// Performs the derivative of a ReLU on each element of the matrix.
//
// Parameters:
//   this - The matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_dReLU(Matrix *this);

// matrix_transpose
// ================
//
// Transposes a matrix.
//
// Parameters:
//   this - The matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_transpose(Matrix *this);

// matrix_subtract
// ===============
//
// Subtracts each element of the other matrix from this matrix.
//
// Parameters:
//    this - The matrix.
//   other - The other matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_subtract(Matrix *this, Matrix *other);

// matrix_multiply_scalar
// ======================
//
// Multiplies each element in a matrix by a scalar value.
//
// Parameters:
//    this - The matrix.
//   value - The scalar to multiply by.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_multiply_scalar(Matrix *this, double value);

// matrix_softmax
// ==============
//
// Performs a softmax operation on each column of the matrix.
//
// Parameters:
//   this - The matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_softmax(Matrix *this);

// matrix_rand
// ===========
//
// Randomizes matrix elements to between -0.5 and 0.5, inclusive.
//
// Parameters:
//   this - The matrix.
void matrix_rand(Matrix *this);

// matrix_clear
// ============
//
// Sets all values in a matrix to 0.
//
// Parameters:
//   this - The matrix.
void matrix_clear(Matrix *this);


#endif // LINALG_H
