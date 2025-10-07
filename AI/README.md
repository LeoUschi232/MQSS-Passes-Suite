# AI Pass Selector

## TODO List

This list is unordered regarding priority.

 Task                                            | Description                                                                                                                                                                                                                                              
-------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Make A2C handle padding on observations         | To vectorize multiple observations into batched observations all observations must be the same length regardless of nr of instructions/gates. This requires padding the shorter observations and the agent must learn to ignore the padded instructions. 
 Create instant-validation of agent on a dataset | After training an agent enable running a process that uses the agent on a dataset, selects and applies the passes for each circuit and prints the depth and instruction count reductions achieved.                                                       
 Optimize convolutional design                   | Read research on designs of convolutional neural networks and make the design of agents with convolutional layers similar to state-of-the-art standardized practices uses.                                                                               
 Create Chemistry Dataset                        | In addition to the MQTBench dataset, create a dataset of quantum checmistry circuits using PySCF and/or OpenFermion.                                                                                                                                     
 Implement A2C using RNN                         | In addition to current convolutional A2C design, create an A2C design which uses RNN and compare its performance to the convolutional design.                                                                                                            
 Implement Experience Replay                     | Implement Experience Replay into the A2C agent.                                                                                                                                                                                                          
 Implement Prioritized Level Replay              | Implement Prioritized Level Replay into the A2C agent.                                                                                                                                                                                                   
 Implement ACER                                  | Implement the Actor-Critic with Experience Replay (ACER) algorithm using convolutional and/or RNN designs.                                                                                                                                               
 Implement A3C agent                             | Implement the Asynchronous Advantage Actor-Critic (A3C) algorithm using convolutional and/or RNN designs.                                                                                                                                                
 Implement PPO agent                             | Implement the Proximal Policy Optimization (PPO) algorithm using convolutional and/or RNN designs.                                                                                                                                                       
 Research and implement Prioritized Level Replay | Research the Prioritized Level Replay technique and implement it in the current agents.                                                                                                                                                                  
 Research and implement StableBaselines3         | Research the StableBaselines3 library and implement it in the current agents.                                                                                                                                                                            

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

### 2. [Different Methods for Optimizing Quantum Computing](ResearchPapers/02_DifferentMethodsForOptimizingQuantumComputing.pdf)

- On average, the agent manages to slightly reduce the gate count, although the depth is slightly increased. Even though
  this agent is, thus, not able to reliably optimize the circuit, it is still able to sometimes achieve an improvement.
  If we, however, train an agent via RL directly on this specific circuit, it does not only learn to reliably improve
  the circuit, but it also finds two further optimizations, each reducing the depth by 1. __[Page 10]__
- The manifold hypothesis is a conjecture of deep learning that posits that neural networks operate by learning low
  dimensional manifolds in high dimensional spaces. A digit that is rotated, translated, or stretched still lies along
  the same manifold. __[Page 79]__

### 3. [Munich Quantum Toolkit and Optimizations of Quantum Programs](ResearchPapers/03_MunichQuantumToolkitAndOptimizationsOfQuantumPrograms.pdf)

- The observations used to guide the reinforcement learning agent are based on seven features—namely the number of
  qubits, the depth of the circuit, and the five composite features of program communication, critical-depth,
  entanglement-ratio, parallelism, and liveness. __[Page 6]__
- Vvarious characteristics are used to describe a quantum circuit for both models: the number of qubits, the depth of
  the circuit, and the five composite features of program communication, critical-depth, entanglement-ratio,
  parallelism, and liveness. __[Page 66]__
- A key challenge in realizing fault-tolerant quantum computers is circuit optimization. Focusing on the most expensive
  gates in fault-tolerant quantum computation, namely the T gates, we address the problem of T-count optimization,
  minimizing the number of T gates that are needed to implement a given circuit. __[Page 77]__

### 4. [Reinforcement Learning Barto Sutton](ResearchPapers/04_ReinforcementLearningBartoSutton.pdf)

- The additional concept that we need is discounting. The agent tries to select actions so that the sum of the
  discounted rewards it receives over the future is maximized. $\gamma$ is a parameter, $0\leq\gamma\leq1$, called the
  discount rate. __[Page 62]__
- The value function of a state $s$ under a policy $\pi$, denoted $v_\pi(s)$, is the expected return when starting
  in $s$ and following $\pi$ thereafter. For Markov Decision Processes, we can define $v_\pi$ formally. Similarly, we
  define the value of taking action $a$ in state $s$ under a policy $\pi$, denoted $q_\pi(s,a)$, as the expected return
  starting from $s$, taking the action $a$, and thereafter following policy $\pi$. We call $q_\pi$ the action-value
  function for policy $\pi$. __[Page 65]__
- The quantity in brackets in the Temporal-Difference update is a sort of error, measuring the difference between the
  estimated value of $S_t$ and the better estimate $R_{t+1}+\gamma V(S_{t+1})$. This quantity, called the
  Temporal-Difference error $\delta_t$, arises in various forms throughout reinforcement learning. __[Page 126]__
- Architecture of a deep convolutional network. This instance was designed to recognize hand-written characters. It
  consists of alternating convolutional and subsampling layers, followed by several fully connected final layers. Each
  convolutional layer produces a number of feature maps- __[Page 229]__
- The $\mathrm{TD}(\lambda)$ algorithm can be understood as one particular way of averaging   $n$-step updates. This
  average
  contains all the $n$-step updates, each weighted proportionally to $\lambda^{n-1}$, where $\lambda\in[0,1]$, and is
  normalized by a factor of $\lambda-1$ to ensure that the weights sum to $1$. The resulting update is toward a return,
  called the $\lambda$-return. __[Page 290]__
- Methods that learn approximations to both policy and value functions are often called actor–critic methods, where
  actor is a reference to the learned policy, and critic refers to the learned value function, usually a state-value
  function. __[Page 322]__
- Only through bootstrapping do we introduce bias and an asymptotic dependence on the quality of the function
  approximation. As we have seen, the bias introduced through bootstrapping and reliance on the state representation is
  often beneficial because it reduces variance and accelerates learning. __[Page 332]__

### 5. [Actor-Critic Reinforcement Learning Frameworks](ResearchPapers/05_ActorCriticReinforcementLearningFrameworks.pdf)

- The objective function for the Actor-Critic algorithm is a combination of the policy gradient for the actor and the
  value function for the critic. $A(s,a)$ is the advantage function representing the advantage of taking the action $a$
  in state $s$. __[Page 6]__
- The Generalized Advantage Estimator $\mathrm{GAE}(\gamma,\lambda)$ is defined as the exponentially-weighted average
  of $k$-step estimators. The advantage estimator has a remarkably simple formula involving a discounted sum of Bellman
  residual terms. There are two notable special cases of this formula, obtained by setting $\lambda=0$
  and $\lambda=1$. __[Page 16]__
- Each policy$\pi$ is represented by a neural network that maps a given state $s$ and goal $g$ to a distribution over
  action $\pi(a|s,g)$. Our policies are trained with PPO using the clipped surrogate objective. We maintain two
  networks, one for the policy $\pi_\theta(a|s,g)$ and another for the value function $V_\psi(s,g)$ with
  parameters $\theta$ and $\psi$ respectively. __[Page 28]__
- Algorithm 1: Proximal Policy Optimization summarizes the common learning procedure used to train all policies. Policy
  updates are performed after a batch of $m=4096$ samples has been collected. Minibatches of size $n=256$ are then
  sampled from the data for each gradient step. A discount factor $\gamma=0.95$ is used for all motions. $\lambda=0.95$
  is used for both $\mathrm{TD}(\lambda)$ and $\mathrm{GAE}(\gamma,\lambda)$. __[Page 39]__
- Asynchronous Advantage Actor-Critic is a classic policy gradient method with the special focus on parallel training.
  In $\mathrm{A3C}$, the critics learn the state-value function $V_w(s)$, while multiple actors are trained in parallel
  and get synced with global parameters from time to time. __[Page 57]__
- Algorithm 2. Actor–Critic with Experience Replay. Estimators mentioned in Steps 6 and 7 are based on the samples in a
  database. Due to more exhaustive exploitation of information experience replay leads to faster learning at the cost of
  additional computation. __[Page 87]__
- <span style="font-variant: small-caps; font-size:15px;">Stable-Baselines3</span> contains the following
  state-of-the-art on-policy and off-policy algorithms, commonly used as experimental baselines: A2C, PPO, DDPG, SAC,
  TD3, HER, and DQN. __[Page 111]__
-

## ML/RL Frameworks

- **LibTorch (PyTorch C++ API)**: Provides a high-level interface similar to PyTorch's Python API, allowing you to build
  models with `torch::nn::Sequential`, `torch::nn::ReLU`, `torch::nn::LeakyReLU`, `torch::nn::Linear`,
  `torch::nn::Conv2d`, etc. It's flexible for custom RL agents via neural nets and supports GPU acceleration.

- **mlpack**: Mature C++ ML toolkit (Armadillo backend) with ANN modules and **built-in RL algorithms** (DQN, Double
  DQN, DDPG, PPO, etc.). Handy if you want both the nets and ready RL baselines in pure C++. Alsoast, scalable C++ ML
  library with an ANN module for feedforward networks (FFN acts like Sequential).
  Supports layers such as `mlpack::ann::Linear`, `mlpack::ann::ReLU`, `mlpack::ann::LeakyReLU`,
  `mlpack::ann::Convolution`, and more. Great for implementing NN-based RL policies without heavy dependencies.

