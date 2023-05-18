#include "numeros.h"

int main(int argc, char **argv)
{
	if (strequ(argv[1], "train"))
	{
		train();
	}
	else if (strequ(argv[1], "test"))
	{
		test();
	}
	else
	{
		image(argv[1]);
	}
}

// train
// =====
//
// Trains the model using the MNIST database.
void train(void)
{
	srand(time(NULL));

	FILE *image_file = fopen("data/train-images.idx3-ubyte", "rb");
	if (image_file == NULL)
	{
		printf("Could not find training images at 'data/train-images.idx3-ubyte'.\n");
		exit(2);
	}

	FILE *label_file = fopen("data/train-labels.idx1-ubyte", "rb");
	if (label_file == NULL)
	{
		printf("Could not find training lables at 'data/train-labels.idx1-ubyte'.\n");
		fclose(image_file);
		exit(2);
	}

	unsigned char *buffer = malloc(50000000);
	if (buffer == NULL)
	{
		printf("Your computer has run out of memory :(\n");
		fclose(image_file);
		fclose(label_file);
		exit(2);
	}

	fread(buffer, 1, 16, image_file);
	fread(buffer, 1, 8, label_file);

	Matrix *pixels = matrix_new(784, BATCH_SIZE);
	unsigned char *labels = malloc(BATCH_SIZE);

	fread(buffer, 784, BATCH_SIZE, image_file);
	fread(labels, 1, BATCH_SIZE, label_file);

	for (unsigned int image = 0; image < BATCH_SIZE; image++)
	{
		for (unsigned int pixel = 0; pixel < 784; pixel++)
		{
			matrix_set(pixels, pixel, image, buffer[image*784 + pixel] / 255.0);
		}
	}

	fclose(image_file);
	fclose(label_file);

	Matrix *W1 = matrix_new(10, 784);
	Matrix *b1 = matrix_new(10, 1);
	Matrix *W2 = matrix_new(10, 10);
	Matrix *b2 = matrix_new(10, 1);
	Matrix *A1, *A2, *Z1, *Z2;
	Matrix *nW1, *nb1, *nW2, *nb2;

	Matrix *answers = matrix_new(10, BATCH_SIZE);
	Matrix *dZ1, *dZ2, *dW1, *dW2, *db1, *db2;
	Matrix *tmp, *tmp2, *tmp3;

	matrix_rand(W1);
	matrix_rand(b1);
	matrix_rand(W2);
	matrix_rand(b2);
	
	double learning_rate = 0.1;

	for (unsigned int i = 1; i <= ITERATIONS; i++)
	{
		tmp = matrix_multiply(W1, pixels);
		Z1 = matrix_add_to_rows(tmp, b1);
		matrix_free(tmp);
		A1 = matrix_ReLU(Z1);
		tmp = matrix_multiply(W2, A1);
		Z2 = matrix_add_to_rows(tmp, b2);
		matrix_free(tmp);
		A2 = matrix_softmax(Z2);

		matrix_clear(answers);
		for (unsigned int image = 0; image < BATCH_SIZE; image++)
		{
			matrix_set(answers, labels[image], image, 1.0);
		}

		dZ2 = matrix_subtract(A2, answers);
		tmp = matrix_transpose(A1);
		dW2 = matrix_multiply(dZ2, tmp);
		matrix_free(tmp);
		db2 = matrix_sum_rows(dZ2);
		tmp = matrix_transpose(W2);
		tmp2 = matrix_multiply(tmp, dZ2);
		tmp3 = matrix_dReLU(Z1);
		dZ1 = matrix_elementwise_multiply(tmp2, tmp3);
		matrix_free(tmp);
		matrix_free(tmp2);
		matrix_free(tmp3);
		tmp = matrix_transpose(dZ1);
		tmp2 = matrix_multiply(pixels, tmp);
		dW1 = matrix_transpose(tmp2);
		matrix_free(tmp);
		matrix_free(tmp2);
		db1 = matrix_sum_rows(dZ1);

		tmp = matrix_multiply_scalar(dW1, learning_rate / BATCH_SIZE);
		nW1 = matrix_subtract(W1, tmp);
		matrix_free(tmp);
		tmp = matrix_multiply_scalar(db1, learning_rate / BATCH_SIZE);
		nb1 = matrix_subtract(b1, tmp);
		matrix_free(tmp);
		tmp = matrix_multiply_scalar(dW2, learning_rate / BATCH_SIZE);
		nW2 = matrix_subtract(W2, tmp);
		matrix_free(tmp);
		tmp = matrix_multiply_scalar(db2, learning_rate / BATCH_SIZE);
		nb2 = matrix_subtract(b2, tmp);
		matrix_free(tmp);

		double mk = 100.0 * mark(A2, labels);

		matrix_free(W1);
		matrix_free(b1);
		matrix_free(W2);
		matrix_free(b2);
		
		matrix_free(A1);
		matrix_free(A2);
		matrix_free(Z1);
		matrix_free(Z2);
		matrix_free(dW1);
		matrix_free(dW2);
		matrix_free(db1);
		matrix_free(db2);
		matrix_free(dZ1);
		matrix_free(dZ2);

		W1 = nW1;
		W2 = nW2;
		b1 = nb1;
		b2 = nb2;

		printf("Training...%.2lf%% Accuracy=%.1lf%%\r", 100.0 * i / ITERATIONS, mk);
		fflush(stdout);
	}
	printf("\n");

	FILE *brainsave = fopen("brainsave", "wb");

	fwrite(W1->data, sizeof(double), 7840, brainsave);
	fwrite(W2->data, sizeof(double), 100, brainsave);
	fwrite(b1->data, sizeof(double), 10, brainsave);
	fwrite(b2->data, sizeof(double), 10, brainsave);

	fclose(brainsave);

	free(buffer);
	free(labels);
	matrix_free(pixels);
	matrix_free(answers);
	matrix_free(W1);
	matrix_free(W2);
	matrix_free(b1);
	matrix_free(b2);
}

// test
// ====
//
// Uses the 'brainsave' file created by train() to test the model's accuracy.
void test(void)
{
	FILE *test_images = fopen("data/t10k-images.idx3-ubyte", "rb");
	if (test_images == NULL)
	{
		printf("Could not find test images at 'data/t10k-images.idx3-ubyte'.\n");
		exit(4);
	}

	FILE *test_labels = fopen("data/t10k-labels.idx1-ubyte", "rb");
	if (test_labels == NULL)
	{
		printf("Could not find test labels at 'data/t10k-labels.idx1-ubyte'.\n");
		fclose(test_images);
		exit(4);
	}

	FILE *brainsave = fopen("brainsave", "rb");
	if (brainsave == NULL)
	{
		printf("Could not find a brainsave file. Run train first.\n");
		fclose(test_images);
		fclose(test_labels);
		exit(4);
	}

	unsigned char *raw_test_data = malloc(TEST_SIZE*1000);
	unsigned char *labels = malloc(TEST_SIZE);

	// Headers.
	fread(raw_test_data, 1, 16, test_images);
	fread(labels, 1, 8, test_labels);

	fread(raw_test_data, 784, TEST_SIZE, test_images);
	fread(labels, 1, TEST_SIZE, test_labels);

	double *W1_buffer = malloc(sizeof(double)*7840);
	double *W2_buffer = malloc(sizeof(double)*100);
	double *b1_buffer = malloc(sizeof(double)*10);
	double *b2_buffer = malloc(sizeof(double)*10);

	fread(W1_buffer, sizeof(double), 7840, brainsave);
	fread(W2_buffer, sizeof(double), 100, brainsave);
	fread(b1_buffer, sizeof(double), 10, brainsave);
	fread(b2_buffer, sizeof(double), 10, brainsave);

	fclose(brainsave);
	fclose(test_images);
	fclose(test_labels);

	Matrix *W1 = matrix_new_from_data(10, 784, W1_buffer);
	Matrix *W2 = matrix_new_from_data(10, 10, W2_buffer);
	Matrix *b1 = matrix_new_from_data(10, 1, b1_buffer);
	Matrix *b2 = matrix_new_from_data(10, 1, b2_buffer);

	Matrix *pixels = matrix_new(784, TEST_SIZE);
	for (unsigned int image = 0; image < TEST_SIZE; image++)
	{
		for (unsigned int pixel = 0; pixel < 784; pixel++)
		{
			matrix_set(pixels, pixel, image, (raw_test_data[image*784 + pixel]) / 255.);
		}
	}
	free(raw_test_data);

	Matrix *A1, *A2, *Z1, *Z2, *tmp;

	tmp = matrix_multiply(W1, pixels);
	Z1 = matrix_add_to_rows(tmp, b1);
	matrix_free(tmp);
	A1 = matrix_ReLU(Z1);
	tmp = matrix_multiply(W2, A1);
	Z2 = matrix_add_to_rows(tmp, b2);
	matrix_free(tmp);
	A2 = matrix_softmax(Z2);

	printf("Accuracy: %.2lf%%.\n", 100.0 * mark(A2, labels));
	free(labels);

	matrix_free(A1);
	matrix_free(A2);
	matrix_free(Z1);
	matrix_free(Z2);
	matrix_free(pixels);
	matrix_free(W1);
	matrix_free(W2);
	matrix_free(b1);
	matrix_free(b2);
}

// image
// =====
//
// Attempts to determine the number contained in a bitmap.
//
// Parameters:
//   path - The path to a 28x28, 24bpp greyscale image.
void image(char *path)
{
	FILE* brainsave = fopen("brainsave", "rb");
	if (brainsave == NULL)
	{
		printf("No brainsave file found. Run train first.\n");
		exit(5);
	}

	unsigned char* raw_pixels = read_image(path);
	Matrix* pixels = matrix_new(784, 1);

	for (unsigned int i = 0; i < 784; i++)
	{
		matrix_set(pixels, i, 0, (255 - raw_pixels[i]) / 255.);
	}
	free(raw_pixels);

	double *W1_buffer = (double*)malloc(sizeof(double) * 7840);
	double *W2_buffer = (double*)malloc(sizeof(double) * 100);
	double *b1_buffer = (double*)malloc(sizeof(double) * 10);
	double *b2_buffer = (double*)malloc(sizeof(double) * 10);

	fread(W1_buffer, sizeof(double), 7840, brainsave);
	fread(W2_buffer, sizeof(double), 100, brainsave);
	fread(b1_buffer, sizeof(double), 10, brainsave);
	fread(b2_buffer, sizeof(double), 10, brainsave);

	Matrix *W1 = matrix_new_from_data(10, 784, W1_buffer);
	Matrix *W2 = matrix_new_from_data(10, 10, W2_buffer);
	Matrix *b1 = matrix_new_from_data(10, 1, b1_buffer);
	Matrix *b2 = matrix_new_from_data(10, 1, b2_buffer);

	Matrix *A1, *A2, *Z1, *Z2, *tmp;

	tmp = matrix_multiply(W1, pixels);
	Z1 = matrix_add_to_rows(tmp, b1);
	matrix_free(tmp);
	A1 = matrix_ReLU(Z1);
	tmp = matrix_multiply(W2, A1);
	Z2 = matrix_add_to_rows(tmp, b2);
	matrix_free(tmp);
	A2 = matrix_softmax(Z2);

	int output = 0;
	double max = 0.0;
	for (int i = 0; i < 10; i++)
	{
		double got = matrix_get(A2, i, 0);
		if (got > max)
		{
			max = got;
			output = i;
		}
	}

	printf("Looks like a %d to me.\n", output);

	matrix_free(A1);
	matrix_free(A2);
	matrix_free(Z1);
	matrix_free(Z2);

	matrix_free(W1);
	matrix_free(W2);
	matrix_free(b1);
	matrix_free(b2);

	fclose(brainsave);
}

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
double mark(Matrix *output, unsigned char *answers)
{
	unsigned int correct = 0;

	for (int image = 0; image < BATCH_SIZE; image++)
	{
		int response = 0;

		for (int i = 0; i < 10; i++)
		{
			if (matrix_get(output, i, image) > matrix_get(output, response, image))
			{
				response = i;
			}
		}

		if (response == answers[image])
		{
			correct++;
		}
	}

	return (double)correct / output->cols;
}
