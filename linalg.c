#include "linalg.h"

#if USE_CUDA
cublasHandle_t cublas;

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
double *matrix_move_to_gpu(Matrix *matrix)
{
	double *gpu;
	int size = sizeof(double) * matrix->rows * matrix->cols;
	cublasStatus_t status = cudaMalloc(&gpu, size);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("Cublas reported an error.\n");
	}

	cudaMemcpy(gpu, matrix->data, size, cudaMemcpyHostToDevice);

	return gpu;
}
#endif

// idx
// ===
//
// Returns the offset of the data at (row,col) in a matrix.
//
// Parameters:
//   this - The matrix containing the data.
//    row - The row.
//    col - The column.
//
// Return:
//   The offset (array index) of the position (row,col).
static unsigned int idx(Matrix *this, double row, double col)
{
	return this->rows * col + row;
}

// mymax
// =====
//
// Returns the maximum of two numbers.
//
// Parameters:
//   a - The first number.
//   b - The second number.
//
// Return:
//   The larger of a or b.
static double mymax(double a, double b)
{
	return (a>b) * a + (a<=b) * b;
}

// matrix_init
// ===========
//
// Must be called before using any of the matrix operations.
void matrix_init(void)
{
	//srand(time(NULL));

#if USE_CUDA
	cublasCreate(&cublas);
#endif
}

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
Matrix *matrix_new(unsigned int rows, unsigned int cols)
{
	Matrix *this = malloc(sizeof(Matrix));

	this->rows = rows;
	this->cols = cols;
	this->data = malloc(sizeof(double) * rows * cols);

	if (this->data == NULL)
	{
		printf("Your computer has run out of memory :(\n");
		exit(3);
	}

	return this;
}

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
Matrix *matrix_new_from_data(unsigned int rows, unsigned int cols, double *data)
{
	Matrix *this = malloc(sizeof(Matrix));

	this->rows = rows;
	this->cols = cols;
	this->data = data;

	return this;
}

// matrix_free
// ===========
//
// Releases the resources used by a matrix.
//
// Parameters:
//   this - The matrix.
void matrix_free(Matrix *this)
{
	free(this->data);
	free(this);
}

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
void matrix_set(Matrix *this, unsigned int row, unsigned int col, double value)
{
	this->data[idx(this, row, col)] = value;
}

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
double matrix_get(Matrix *this, unsigned int row, unsigned int col)
{
	return this->data[idx(this, row, col)];
}

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
#if USE_CUDA
Matrix *matrix_multiply(Matrix *this, Matrix *other, cublasOperation_t matrix_operation, cublasOperation_t other_operation)
#else
Matrix *matrix_multiply(Matrix *this, Matrix *other)
#endif
{
	bool transpose_matrix = false, transpose_other = false;
#if USE_CUDA
	transpose_matrix = matrix_operation == CUBLAS_OP_T;
	transpose_other = other_operation == CUBLAS_OP_T;
#endif

	unsigned int matcols = (transpose_matrix) ? this->rows : this->cols;
	unsigned int othrows = (transpose_other) ? other->cols : other->rows;

	if (matcols != othrows)
	{
		printf("Cannot multiply matrices due to incompatible sizes: (%u,%u) and (%u,%u).\n", this->rows, this->cols, other->rows, other->cols);
		exit(1);
	}

	Matrix *output = matrix_new((transpose_matrix) ? this->cols : this->rows, (transpose_other) ? other->rows : other->cols);

#if USE_CUDA
	double alpha = 1, beta = 0;

	double *matrix_gpu = matrix_move_to_gpu(this);
	double *other_gpu = matrix_move_to_gpu(other);

	int m = (transpose_matrix) ? this->cols : this->rows;
	int n = (transpose_other) ? other->rows : other->cols;
	int k = (transpose_matrix) ? this->rows : this->cols;

	double *output_gpu;
	int output_gpu_size = sizeof(double) * m * n;
	cudaMalloc(&output_gpu, output_gpu_size);

	cublasStatus_t status = cublasDgemm(
		cublas,
		matrix_operation, other_operation,
		m, n, k,
		&alpha,
		matrix_gpu, this->rows,
		other_gpu, other->rows,
		&beta,
		output_gpu, output->rows
	);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("Cublas reported an error.\n");
		exit(5);
	}

	cudaMemcpy(output->data, output_gpu, output_gpu_size, cudaMemcpyDeviceToHost);

	cudaFree(matrix_gpu);
	cudaFree(other_gpu);
	cudaFree(output_gpu);

#else

	double sum;

	for (unsigned int thisRow = 0; thisRow < this->rows; thisRow++)
	{
		for (unsigned int otherCol = 0; otherCol < other->cols; otherCol++)
		{
			sum = 0.0;

			for (unsigned int i = 0; i < this->cols; i++)
			{
				sum += matrix_get(this, thisRow, i) * matrix_get(other, i, otherCol);
			}

			matrix_set(output, thisRow, otherCol, sum);
		}
	}
	
#endif

	return output;
}

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
Matrix *matrix_elementwise_multiply(Matrix *this, Matrix *other)
{
	if (this->rows != other->rows || this->cols != other->cols)
	{
		printf("Cannot multiply matrices due to incompatible sizes.\n");
		exit(1);
	}

	Matrix *output = matrix_new(this->rows, this->cols);

	for (unsigned int row = 0; row < this->rows; row++)
	{
		for (unsigned int col = 0; col < this->cols; col++)
		{
			matrix_set(output, row, col, matrix_get(this, row, col) * matrix_get(other, row, col));
		}
	}

	return output;
}

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
Matrix *matrix_add_to_rows(Matrix *this, Matrix *other)
{
	if (this->rows != other->rows)
	{
		printf("Cannot add matrices due to incompatible sizes.\n");
		exit(1);
	}

	Matrix *output = matrix_new(this->rows, this->cols);

	for (int col = 0; col < this->cols; col++)
	{
		for (int row = 0; row < this->rows; row++)
		{
			matrix_set(output, row, col, matrix_get(this, row, col) + matrix_get(other, row, 0));
		}
	}

	return output;
}

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
Matrix *matrix_sum_rows(Matrix *this)
{
	Matrix *output = matrix_new(this->rows, 1);
	double sum;

	for (unsigned int row = 0; row < this->rows; row++)
	{
		sum = 0.0;

		for (unsigned int col = 0; col < this->cols; col++)
		{
			sum += matrix_get(this, row, col);
		}

		matrix_set(output, row, 0, sum);
	}

	return output;
}

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
Matrix *matrix_ReLU(Matrix *this)
{
	Matrix *output = matrix_new(this->rows, this->cols);

	for (unsigned int row = 0; row < this->rows; row++)
	{
		for (unsigned int col = 0; col < this->cols; col++)
		{
			matrix_set(output, row, col, mymax(matrix_get(this, row, col), 0));
		}
	}

	return output;
}

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
Matrix *matrix_dReLU(Matrix *this)
{
	Matrix *output = matrix_new(this->rows, this->cols);

	for (unsigned int row = 0; row < this->rows; row++)
	{
		for (unsigned int col = 0; col < this->cols; col++)
		{
			matrix_set(output, row, col, matrix_get(this, row, col) > 0);
		}
	}

	return output;
}

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
Matrix *matrix_transpose(Matrix *this)
{
	Matrix *output = matrix_new(this->cols, this->rows);

	for (unsigned int row = 0; row < this->rows; row++)
	{
		for (unsigned int col = 0; col < this->cols; col++)
		{
			matrix_set(output, col, row, matrix_get(this, row, col));
		}
	}

	return output;
}

// matrix_subtract
// ===============
//
// Subtracts each element of the other matrix from this matrix.
//
// Parameters:
//    this - The matrix.
//   other - The other matrix.
//   scale - The amount by which to scale the other matrix.
//
// Return:
//   A newly allocated matrix. Call matrix_free() when no longer needed.
Matrix *matrix_subtract(Matrix *this, Matrix *other, double scale)
{
	if (this->cols != other->cols || this->rows != other->rows)
	{
		printf("Cannot subtract matrices due to incompatible sizes.\n");
		exit(1);
	}

	Matrix *output = matrix_new(this->rows, this->cols);

	for (unsigned int row = 0; row < this->rows; row++)
	{
		for (unsigned int col = 0; col < this->cols; col++)
		{
			matrix_set(output, row, col, matrix_get(this, row, col) - matrix_get(other, row, col) * scale);
		}
	}

	return output;
}

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
Matrix *matrix_multiply_scalar(Matrix *this, double value)
{
	Matrix *output = matrix_new(this->rows, this->cols);

	for (unsigned int row = 0; row < this->rows; row++)
	{
		for (unsigned int col = 0; col < this->cols; col++)
		{
			matrix_set(output, row, col, matrix_get(this, row, col) * value);
		}
	}

	return output;
}

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
Matrix *matrix_softmax(Matrix *this)
{
	Matrix *output = matrix_new(this->rows, this->cols);

	for (unsigned int col = 0; col < this->cols; col++)
	{
		double sum = 0.0;
		for (unsigned int row = 0; row < this->rows; row++)
		{
			sum += exp(matrix_get(this, row, col));
		}

		for (unsigned int row = 0; row < this->rows; row++)
		{
			matrix_set(output, row, col, exp(matrix_get(this, row, col)) / sum);
		}
	}

	return output;
}

// matrix_rand
// ===========
//
// Randomizes matrix elements to between -0.5 and 0.5, inclusive.
//
// Parameters:
//   this - The matrix.
void matrix_rand(Matrix *this)
{
	for (int row = 0; row < this->rows; row++)
	{
		for (int col = 0; col < this->cols; col++)
		{
			matrix_set(this, row, col, (double)rand() / RAND_MAX - 0.5);
		}
	}
}

// matrix_clear
// ============
//
// Sets all values in a matrix to 0.
//
// Parameters:
//   this - The matrix.
void matrix_clear(Matrix *this)
{
	for (int row = 0; row < this->rows; row++)
	{
		for (int col = 0; col < this->cols; col++)
		{
			matrix_set(this, row, col, 0);
		}
	}
}

// matrix_print
// ============
//
// Prints the given matrix in a clean format.
//
// Parameters:
//   matrix - The matrix to print.
void matrix_print(Matrix *matrix)
{
	unsigned int row = 0, col;

	while (row < matrix->rows)
	{
		printf("[ ");
		col = 0;
		while (col < matrix->cols)
		{
			printf("%7.2lf", matrix_get(matrix, row, col));

			if (matrix->cols > 10 && col == 6)
			{
				col = matrix->cols - 6;
				printf(" ... ");
			}
			else
			{
				col++;
			}
		}
		printf(" ]\n");

		if (matrix->rows > 10 && row == 6)
		{
			row = matrix->rows - 6;
			printf(".\n.\n.\n");
		}
		else
		{
			row++;
		}
	}

	printf("\n\n\n");
}