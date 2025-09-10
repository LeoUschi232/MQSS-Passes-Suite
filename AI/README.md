# AI Pass Selector

## Build

The AI module uses large external libraries that will not be pushed to github.
Before building this module, you will have to download, install and link all external tools to this project.
This guide assumes the project runs inside a docker container in the folder `/workspace`.
Given that, the absolute path of the `AI` subdirectory should be `/workspace/AI`.
Should your project structre differ, you will have to adjust the filepaths accordingly.
In that case replace all instances of `/workspace/AI` with your absolute path to the `AI` subdirectory.
Run the following commands:

```shell
cd /workspace/AI/external
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
rm -rf libtorch-shared-with-deps-latest.zip
```

```shell
cd /workspace/AI/external
git clone https://github.com/leggedrobotics/tensorflow-cpp.git
cd tensorflow-cpp/eigen
./install.sh --run-cmake
cd ../tensorflow
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/workspace/AI/external -DCMAKE_BUILD_TYPE=Release ..
make install -j
cd /workspace/AI/external
rm -rf tensorflow-cpp
```

## ML/RL Frameworks

### Lightweight neural-net libraries (header-only or minimal deps)

* **tiny-dnn** — Header-only, straightforward `sequential` nets (Linear/Conv/ReLU/LeakyReLU). CPU-centric; fine for
  small/embedded RL or classic control.
* **dlib DNN** — Modern C++ with a compile-time graph; supports Conv/FC and common activations. Good CPU performance;
  some CUDA support. Nice for compact models inside C++ apps.
* **mlpack** — Mature C++ ML toolkit (Armadillo backend) with ANN modules and **built-in RL algorithms** (DQN, Double
  DQN, DDPG, PPO, etc.). Handy if you want both the nets and ready RL baselines in pure C++.

### Inference-first (use for deployment or if you train elsewhere)

* **ONNX Runtime (C++ API)** — Load models exported from PyTorch/TF. Fast CPU/GPU backends. You won’t “build” layers
  here; you run exported graphs—perfect for C++ RL inference loops with training done offline.
* **NVIDIA TensorRT** — High-performance GPU inference. Not for defining layers ad-hoc; you import/optimize a trained
  model. Great when your RL agent must act at ultra-low latency.
* **Intel OpenVINO** — Excellent CPU/iGPU inference on Intel hardware; similar use case to ONNX/TensorRT.

### Build-your-own (tensors + autodiff pieces)

* **Eigen or xtensor + autodiff (Adept, CppAD, Stan Math)** — If you want to craft a tiny `Sequential` with Linear/Conv
  and custom activations yourself. Max control, minimal dependencies, but you’ll implement training loops and CUDA
  paths.
* **tiny-cuda-nn** — CUDA-focused tiny MLPs (hash grids, etc.). Niche but blazing fast for specific architectures if
  your RL uses compact networks.

### What to pick

* **Want seamless C++ training + flexible RL?** Go **LibTorch**.
* **Need pure C++ with included RL algorithms/baselines?** **mlpack**.
* **Already train in Python but deploy C++?** **ONNX Runtime** (CPU/GPU) or **TensorRT** (NVIDIA).
* **Tiny footprint / embedded?** **tiny-dnn** or **dlib**.
* **Maximum control / researchy kernels?** Eigen/xtensor + an autodiff lib.

### Notes for your stack

* All of the above work well on Arch Linux or inside an Ubuntu container via **CMake** (+ **vcpkg** or **Conan** if you
  like).
* For RL environments: if you need Gym-style envs in C++, many teams glue to Python envs with **pybind11** while keeping
  the agent in C++; otherwise mlpack’s examples and custom C++ envs are a good start.

If you want, tell me your target (CPU/GPU, latency vs. training speed, dependency tolerance), and I’ll sketch a minimal
CMake project with your chosen library and a `Sequential{Linear→ReLU→Linear}` policy stub.

Here’s a short list of top C++ frameworks and libraries suitable for machine learning (and by extension, reinforcement
learning implementations), focusing on those that allow seamless neural network construction with features like
sequential models, activation functions (e.g., ReLU, LeakyReLU), linear/dense layers, convolutional layers, and more.
These are selected based on their performance, ease of use for NNs, and community support. I've prioritized pure C++
options or those with strong C++ APIs.

- **LibTorch (PyTorch C++ API)**: Provides a high-level interface similar to PyTorch's Python API, allowing you to build
  models with `torch::nn::Sequential`, `torch::nn::ReLU`, `torch::nn::LeakyReLU`, `torch::nn::Linear`,
  `torch::nn::Conv2d`, etc. It's flexible for custom RL agents via neural nets and supports GPU acceleration.

- **mlpack**: A fast, scalable C++ ML library with an ANN module for feedforward networks (FFN acts like Sequential).
  Supports layers such as `mlpack::ann::Linear`, `mlpack::ann::ReLU`, `mlpack::ann::LeakyReLU`,
  `mlpack::ann::Convolution`, and more. Great for implementing NN-based RL policies without heavy dependencies.

- **tiny-dnn**: A lightweight, header-only C++ library for deep learning. Enables sequential models via
  `network<sequential>`, with layers like `conv` (convolutional), `fc` (fully connected/linear), `relu`, `leaky_relu`,
  etc. Ideal for quick prototyping of NNs for RL on resource-constrained setups.

- **OpenNN**: An open-source C++ library focused on high-performance neural networks. Supports building multilayer
  perceptrons with activations like ReLU (rectified linear units), along with linear and other layers; suitable for
  regression/classification tasks that can extend to RL value functions.

- **Caffe**: A deep learning framework in C++ optimized for speed, especially in vision tasks. Allows defining networks
  via prototxt configs or code, with support for convolutional layers, ReLU/LeakyReLU activations, inner product (
  linear) layers, etc. Useful for CNN-based RL environments.



