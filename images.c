#include "images.h"

static uint32_t readint(int bytecount, unsigned char* data)
{
	if (bytecount == 4)
	{
		uint32_t value;
		unsigned char* ptr = (unsigned char*)&value;
		ptr[0] = data[0];
		ptr[1] = data[1];
		ptr[2] = data[2];
		ptr[3] = data[3];

		return value;
	}
	else if (bytecount == 2)
	{
		uint16_t value;
		unsigned char* ptr = (unsigned char*)&value;
		ptr[0] = data[0];
		ptr[1] = data[1];

		return value;
	}

	return 0;
}

unsigned char* read_image(char* path)
{
	FILE* file = fopen(path, "rb");
	if (file == NULL)
	{
		printf("No file '%s' found.\n", path);
		exit(5);
	}

	struct _stat metadata;
	if (_stat(path, &metadata) == -1)
	{
		printf("Error getting file metadata.\n");
		fclose(file);
		exit(5);
	}

	unsigned char* bytes = (unsigned char*)malloc(metadata.st_size);
	if (bytes == NULL)
	{
		printf("Computer is out of memory.\n");
		fclose(file);
		exit(5);
	}

	fread(bytes, 1, metadata.st_size, file);
	fclose(file);

	if (!(bytes[0] == 'B' && bytes[1] == 'M'))
	{
		printf("The file '%s' is not a bitmap.\n", path);
		free(bytes);
		exit(5);
	}

	uint32_t width = readint(4, bytes + 18);
	uint32_t height = readint(4, bytes + 22);
	uint16_t bits_per_pixel = readint(2, bytes + 28);
	uint32_t compression_method = readint(4, bytes + 30);
	uint32_t offset = readint(4, bytes + 10);

	if (bits_per_pixel != 24)
	{
		printf("Only 24 bits per pixel images are supported.\n");
		free(bytes);
		exit(5);
	}

	if (compression_method != 0)
	{
		printf("Only uncompressed images are supported.\n");
		free(bytes);
		exit(5);
	}

	if (!(width == 28 && height == 28))
	{
		printf("The image must be 28 by 28 pixels.\n");
		free(bytes);
		exit(5);
	}

	unsigned int rowstride = width * 3;
	if (rowstride & 0b11)
	{
		rowstride = (rowstride + 4) & 0xFFFFFFF0;
	}

	unsigned char* output = (unsigned char*)malloc(width * height);
	if (output == NULL)
	{
		printf("Your computer is out of memory.\n");
		free(bytes);
		exit(5);
	}

	for (unsigned int row = 0; row < height; row++)
	{
		for (unsigned int col = 0; col < width; col++)
		{
			output[(height-row-1) * width + col] = bytes[offset + row * rowstride + (col * 3)];
		}
	}

	free(bytes);
	return output;
}