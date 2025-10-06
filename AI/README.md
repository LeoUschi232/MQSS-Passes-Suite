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

## Research Paper Notes

### 1. [Optimizing Classical Compiler Passes](ResearchPapers/01_OptimizingClassicalCompilerPasses.pdf)

- Machine learning scales to large corpora of training examples, which we expect to increase the likelihood of obtaining
  policies that generalize well. This is important because, we do not want to retrain policies too frequently. It is an
  adoption blocker. __[Page 8]__
- Near optimal pass sequence can be directly predicted with high accuracy via Graph Edge Attention Network (GEAN), a
  graph neural network (GNN) architecture. RL-based approaches like PPO operating on the original compiler pass space
  often suffer from unstable training due to inaccurate value estimation and sparse reward space and fail to generalize
  to unseen programs at inference. As a contender to GNN, use transformers to process graphs. If we effectively encode
  positional and local sub-structures of graphs and feed them to the transformer, then the transformer can outperform
  the classical GNN models.  __[Page 30]__
- Markov chain oracle outperforms the Independent Identically Distributed probability distribution oracle, followed by
  the RIC methodology using a uniform probability distribution. __[Page 120]__
- 
