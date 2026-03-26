# Artificial Neural Network from scratch

A Rust implementation of a basic artificial neural network trained on MNIST data, featuring matrix operations and visualization with Raylib.

## Dependencies

Raylib in rust depends on `cmake` and `glfw`

## Development log

First iteration of the matrix library returned a newly allocated matrix for each operation. This takes so long that I haven't had the patience to even let the program run an entire training data epoch by itself, on my work laptop with a high end intel core ultra 7 165H, it took 20s to train a single 784x200x10 network over 1000 training examples.

Changing this to be all inplace operations over preallocated matrices.

## References

https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/

Mnist dataset - format:
https://github.com/cvdfoundation/mnist
