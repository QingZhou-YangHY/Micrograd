# Micrograd

A tiny autograd engine implementation based on [Andrej Karpathy's tutorial](https://github.com/karpathy/micrograd), designed to help understand the underlying principles of automatic differentiation and how advanced frameworks like PyTorch work internally.

## Difference with PyTorch

Micrograd is a scalar-valued engine. But in PyTorch, everything is based around tensors. Tensors are just n-dimensional arrays of scalars.