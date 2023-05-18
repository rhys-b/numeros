#include "linalg.h"

static unsigned int idx(Matrix *this, double row, double col)
{
	return this->cols * row + col;
}

static double mymax(double a, double b)
{
	return (a>b) * a + (a<=b) * b;
}

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

Matrix *matrix_new_from_data(unsigned int rows, unsigned int cols, double *data)
{
	Matrix *this = malloc(sizeof(Matrix));

	this->rows = rows;
	this->cols = cols;
	this->data = data;

	return this;
}

void matrix_free(Matrix *this)
{
	free(this->data);
	free(this);
}

void matrix_set(Matrix *this, unsigned int row, unsigned int col, double value)
{
	this->data[idx(this, row, col)] = value;
}

double matrix_get(Matrix *this, unsigned int row, unsigned int col)
{
	return this->data[idx(this, row, col)];
}

Matrix *matrix_multiply(Matrix *this, Matrix *other)
{
	if (this->cols != other->rows)
	{
		printf("Cannot multiply matrices due to incompatible sizes.\n");
		exit(1);
	}

	Matrix *output = matrix_new(this->rows, other->cols);
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

	return output;
}

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

Matrix *matrix_subtract(Matrix *this, Matrix *other)
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
			matrix_set(output, row, col, matrix_get(this, row, col) - matrix_get(other, row, col));
		}
	}

	return output;
}

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
