# AI Pass Selector

## ML/RL Frameworks

- **LibTorch (PyTorch C++ API)**: Provides a high-level interface similar to PyTorch's Python API, allowing you to build
  models with `torch::nn::Sequential`, `torch::nn::ReLU`, `torch::nn::LeakyReLU`, `torch::nn::Linear`,
  `torch::nn::Conv2d`, etc. It's flexible for custom RL agents via neural nets and supports GPU acceleration.

- **mlpack**: Mature C++ ML toolkit (Armadillo backend) with ANN modules and **built-in RL algorithms** (DQN, Double
  DQN, DDPG, PPO, etc.). Handy if you want both the nets and ready RL baselines in pure C++. Alsoast, scalable C++ ML
  library with an ANN module for feedforward networks (FFN acts like Sequential).
  Supports layers such as `mlpack::ann::Linear`, `mlpack::ann::ReLU`, `mlpack::ann::LeakyReLU`,
  `mlpack::ann::Convolution`, and more. Great for implementing NN-based RL policies without heavy dependencies.


