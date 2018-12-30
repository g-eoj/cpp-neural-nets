# Neural Networks in C++

Neural networks in C++, using the Eigen 3 template library for linear algebra.

Implemented so far:
- Dense layers with ReLU or Softmax activation. 
- Dropout layers.
- Neural net that can be initialized with an arbitrary number of layers.
- SGD optimizer with momentum.
- Preprocessing and evaluation utilities.
- `main.cpp` is currently hardcoded to train and validate a two layer net on `wine.data` from  https://archive.ics.uci.edu/ml/datasets/wine.
- `tests.cpp` checks the implementation of layers, neural networks, and utilities.
