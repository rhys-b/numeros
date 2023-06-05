# Numeros

A classic first machine learning excersize for recognizing hand written digits,
trained using the legends over at MNIST, and written entirely in C,
using only the C standard library and Nvidia's matrix multiplication.

## Using

The model originally took a long time to train, because I only used 'vanilla' (unoptimized) matrix
multipliation. Therefore, I added CUDA support (Nvidia's GPU library) to speed things up.

The model is only able to get about 85% accuracy on the test data.

All bitmaps given must be 24 bits per pixel, and 28 by 28 pixels, black on white.

## Compiling

A C compiler is required. For compiling with CUDA, I
highly recommend using Nvidia's NVCC compiler, as it takes care of
many of the dependencies. Look [here](https://developer.nvidia.com/how-to-cuda-c-cpp)
for how to get started with CUDA.

After cloning the repository, compile the model with

```
gcc -o numeros numeros.c linalg.c images.c
```

to only use the C standard library. `gcc` can be substituted for any C compiler of your choice.
To use Nvidia GPU accelerated computing (if your computer has Nvidia graphics), use

```
nvcc -o numeros numeros.c linalg.c images.c -DUSE_CUDA=1 -lcublas
```

On windows, change `-lcublas` to `-lcublas.lib`.

Train the model using

```
./numeros train
```

After training, test the model using

```
./numeros test
```

After training, try a bitmap image on the model using

```
./numeros <file_path>
```
