# Numeros

A classic first machine learning excersize for recognizing hand written digits,
trained using the legends over at MNIST, and written entirely in C,
using only the C standard library.

## Using

The model takes a long time to train, because I only used 'vanilla' (unoptimized) matrix
multipliation, and also only gets about 85% accuracy on the test data.

All bitmaps given must be 24 bits per pixel, and 28 by 28 pixels, black on white.

## Compiling

A C compiler is required. The Mac and Linux examples use `gcc`,
and the Windows examples use `cl`.

### Linux/Mac
After cloning the repository, compile the model with

```
gcc -o numeros numeros.c linalg.c images.c
```

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

### Windows
After cloning the repository, compile the model with

```
cl numeros.c linalg.c images.c
```

Train the model using

```
numeros.exe train
```

After training, test the model using

```
numeros.exe test
```

After training, try a bitmap on the model using

```
numeros.exe <file_path>
```
