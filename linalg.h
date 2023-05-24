#ifndef LINALG_H
#define LINALG_H


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#if USE_CUDA
#include <cublas_v2.h>
#endif


typedef struct
{
	unsigned int rows, cols;
	double *data;
} Matrix;

#if USE_CUDA
// matrix_move_to_gpu
// ==================
//
// Moves memory from a matrix's data into GPU dedicated memory.
//
// Parameters:
//   matrix: The matrix whose data is to be moved.
//
// Return:
//   The pointer to the GPU device's memory location.
double *matrix_move_to_gpu(Matrix *matrix);
#endif

// matrix_init
// ===========
//
// Must be called before using any of the matrix operations.
void matrix_init(void);

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
//   data - The left to right then down data of the matrix. The matrix takes ownership of matrix data.
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
//   matrix - The matrix.
void matrix_free(Matrix *matrix);

// matrix_set
// ==========
//
// Sets a value in a matrix.
//
// Parameters:
//   matrix - The matrix.
//      row - The row.
//      col - The column.
//    value - The value to set the element to.
void matrix_set(Matrix *matrix, unsigned int row, unsigned int col, double value);

// matrix_get
// ==========
//
// Gets a value from a matrix.
//
// Parameters:
//   matrix - The matrix.
//    row - The row.
//    col - The column.
//
// Return:
//   The value in the matrix.
double matrix_get(Matrix *matrix, unsigned int row, unsigned int col);

// matrix_multiply
// ===============
//
// Multiplies matrix matrix with another matrix.
//
// Parameters:
//    matrix - The first matrix.
//   other - The second matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
#if USE_CUDA
Matrix *matrix_multiply(Matrix *matrix, Matrix *other, cublasOperation_t matop, cublasOperation_t othop);
#else
Matrix *matrix_multiply(Matrix *matrix, Matrix *other);
#endif

// matrix_elementwise_multiply
// ===========================
//
// Multiplies each element of a matrix with the corresponding element of another matrix.
//
// Parameters:
//    matrix - The first matrix.
//   other - The second matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_elementwise_multiply(Matrix *matrix, Matrix *other);

// matrix_add_to_rows
// ==================
//
// Adds the value in the first column of the other matrix to each element in the corresponding row of the first matrix.
//
// Parameters:
//    matrix - The first matrix.
//   other - The second matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_add_to_rows(Matrix *matrix, Matrix *other);

// matrix_sum_rows
// ===============
//
// Creates an (N,1) matrix where each element is the sum of the row of the input matrix.
//
// Parameters:
//   matrix - The matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_sum_rows(Matrix *matrix);

// matrix_ReLU
// ===========
//
// Performs a ReLU on each element of the matrix.
//
// Parameters:
//   matrix - The matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_ReLU(Matrix *matrix);

// matrix_dReLU
// ============
//
// Performs the derivative of a ReLU on each element of the matrix.
//
// Parameters:
//   matrix - The matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_dReLU(Matrix *matrix);

// matrix_transpose
// ================
//
// Transposes a matrix.
//
// Parameters:
//   matrix - The matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_transpose(Matrix *matrix);

// matrix_subtract
// ===============
//
// Subtracts each element of the other matrix from matrix matrix.
//
// Parameters:
//    matrix - The matrix.
//    other - The other matrix.
//    scale - The amount by which to scale the other matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_subtract(Matrix *matrix, Matrix *other, double scale);

// matrix_multiply_scalar
// ======================
//
// Multiplies each element in a matrix by a scalar value.
//
// Parameters:
//    matrix - The matrix.
//    value - The scalar to multiply by.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_multiply_scalar(Matrix *matrix, double value);

// matrix_softmax
// ==============
//
// Performs a softmax operation on each column of the matrix.
//
// Parameters:
//   matrix - The matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_softmax(Matrix *matrix);

// matrix_rand
// ===========
//
// Randomizes matrix elements to between -0.5 and 0.5, inclusive.
//
// Parameters:
//   matrix - The matrix.
void matrix_rand(Matrix *matrix);

// matrix_clear
// ============
//
// Sets all values in a matrix to 0.
//
// Parameters:
//   matrix - The matrix.
void matrix_clear(Matrix *matrix);

// matrix_print
// ============
//
// Prints the given matrix in a clean format.
//
// Parameters:
//   matrix - The matrix to print.
void matrix_print(Matrix *matrix);


#endif // LINALG_H