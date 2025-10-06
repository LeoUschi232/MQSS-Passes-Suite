# AI Prompts

## Grok Prompts

The current setup pads all observations to the maximum number of instructions (`maxN`) across the batch, using
zero-padding for shorter sequences. This means the agent processes zero-vectors for padded instructions, which may
dilute the signal during convolution (since `Conv1d` mixes features across windows that could include padded regions).
While zero-padding is neutral and contributes nothing directly to the convolution sums, it can still indirectly affect
the representation if the kernel slides over boundaries between valid and padded instructions. To ensure the agent
truly "sees" only the non-padded instructions (without processing large blocks of trailing zeros), you can switch to
ragged tensors (nested tensors in PyTorch terminology). This handles variable-length sequences natively without
batch-level padding.

### Why Ragged/Nested Tensors?

- Your problem is level-1 raggedness: The batch is ragged only along one axis (number of instructions `N_i` per
  circuit/environment), while the feature size (`IRP = MAX_QUBITS_TO_IRP(max_qubits)`) is fixed.
- Nested tensors allow you to batch tensors of shape `[N_i, IRP]` without padding to a fixed `maxN`. The resulting
  nested tensor implicitly represents a batch with unbound dimensions along the ragged axis.
- This avoids adding trailing zero-vectors entirely. The only "padding" will be the edge padding specified in your
  `Conv1d` layers (e.g., `padding = (max_qubits - 1) / 2`), which is applied temporarily during convolution to handle
  boundaries—standard for any sequence model and not the same as batch padding.

### Compatibility with `Conv1d`

- PyTorch's `torch::nn::Conv1d` (and `torch.nn.functional.conv1d`) supports nested tensors natively (since PyTorch
  1.12+). It applies the convolution component-wise to each tensor in the nest.
    - Input: A nested tensor where each component is `[in_channels, L_i]` (here, `in_channels = IRP` or similar after
      transposition, `L_i = N_i`).
    - Output: A new nested tensor where each component is `[out_channels, L_i']`, with `L_i'` computed based on kernel
      size, stride, dilation, and padding (approximately `N_i` in your case, since you're using "same" padding to
      preserve length).
    - If `N_i < kernel_size` for some sequences, the convolution will still work as long as your fixed padding ensures a
      valid output size (e.g., no negative lengths). With your padding ≈ `(kernel_size - 1) / 2`, output length ≈ input
      length for all `N_i`, even small ones.
- Other built-in layers like `nn::AdaptiveAvgPool1d`, `nn::LayerNorm`, `nn::Flatten`, and `nn::Softmax` also support
  nested tensors (they apply component-wise).
- Custom layers (e.g., `TransposeContiguous`, `HalfScalingLayer`) must be modified to support nested inputs, as shown
  below. Without this, they'll fail on nested tensors.
- Limitations:
    - Operations like `multinomial`, `log`, `gather`, and `sum` do **not** natively support nested tensors (they expect
      regular tensors).
    - After the model (where outputs are fixed-size per component, e.g., `[NR_PASSES]` for the actor), you'll unbind the
      nested tensor into a vector of components and stack them into a regular tensor `[B, NR_PASSES]`.
    - Nested tensors are less efficient on GPU than padded batches (no vectorized operations across the batch dim), but
      with small `B` (`nr_parallel_environments` typically ≤ 64), the impact is minimal.
    - Ensure your PyTorch version ≥ 1.13 for stable nested tensor support in C++.

### Implementation Steps

1. **Modify `ParallelEnvironments` to Return Nested Tensors**:

- Update `get_batched_observations_with_padding()` to return only the nested tensor (no mask needed anymore).
- Don't pad individual `InstructionsTensor`s. Instead, collect unpadded tensors into a `std::vector<torch::Tensor>` and
  create a nested tensor.
- Update `get_batched_observations()` accordingly (remove the mask return).

```C++
#include <torch/nested_tensor.h>  // Ensure this is included

std::pair<torch::Tensor, torch::Tensor>
ParallelEnvironments::get_batched_observations_with_padding() const {
   // ... (existing code to get observations vector)

   std::vector<torch::Tensor> obs_list;
   obs_list.reserve(B);
   unsigned int maxN = 0u;  // Still compute for logging/debug if needed
   int64_t IRP = 0u;

   for (int64_t batch = 0; batch < B; batch++) {
       InstructionsTensor<double> instruction_tensor = observations[batch];
       unsigned int N = instruction_tensor.shape[0];  // Actual N, no pad
       maxN = std::max(maxN, N);
       if (IRP <= 0) {
           IRP = instruction_tensor.shape[1];
       } else if (IRP != instruction_tensor.shape[1]) {
           throw std::runtime_error("Inconsistent IRP.");
       }
       torch::Tensor tensor = torch::from_blob(instruction_tensor.raw(), {N, IRP}, options).clone();
       obs_list.emplace_back(std::move(tensor));
   }

   if (obs_list.empty()) {
       // Handle empty case (rare)
       return {torch::zeros({B, 1, IRP}, options), torch::ones({B, 1}, options)};
   }

   // Create nested tensor: Shape unbound in dim=0 (N_i), bound in dim=1 (IRP)
   torch::Tensor batched_observations = torch::nested_tensor(obs_list);

   // Return dummy mask for compatibility (or refactor callers to not expect it)
   torch::Tensor dummy_mask = torch::empty({0}, options);  // Or remove entirely
   return {batched_observations, dummy_mask};
}

torch::Tensor ParallelEnvironments::get_batched_observations() const {
   auto [batched, _] = get_batched_observations_with_padding();
   return batched;
}
```

2. **Modify Custom Modules to Support Nested Tensors**:

- For each custom module (e.g., `TransposeContiguous`, `HalfScalingLayer`), update `forward` to recursively handle
  nested inputs using `torch::nested::map_nested_tensor`.
- This applies the operation component-wise to regular (non-nested) tensors in the nest.
- Built-in modules like `Conv1d` already handle this.

Example for `TransposeContiguous` (assume it's a custom `torch::nn::Module`):

```C++
struct TransposeContiguous : public torch::nn::Module {
   int dim0, dim1;
   TransposeContiguous(int d0, int d1) : dim0(d0), dim1(d1) {}

   torch::Tensor forward(torch::Tensor input) {
       if (input.is_nested()) {
           auto map_fn = [this](torch::Tensor t) -> torch::Tensor {
               return this->forward(t);  // Recurse: t is regular
           };
           return torch::nested::map_nested_tensor(map_fn, input);
       }
       // Normal operation for regular tensor
       return input.transpose(dim0, dim1).contiguous();
   }
 };
```

Do the same for `HalfScalingLayer` (placeholder assuming it halves the input):

```C++
struct HalfScalingLayer : public torch::nn::Module {
   torch::Tensor forward(torch::Tensor input) {
       if (input.is_nested()) {
           auto map_fn = [this](torch::Tensor t) -> torch::Tensor {
               return this->forward(t);
           };
           return torch::nested::map_nested_tensor(map_fn, input);
       }
       // Original implementation
       return input / 2.0;
   }
};
```

- In `A2C_CONV2` constructor, ensure the `Sequential` uses these updated custom modules. The dims in
  `TransposeContiguous` should be `(0, 1)` (since components are `[N_i, IRP]`, transpose to `[IRP, N_i]`).

3. **Modify `BaseA2CAgent` to Handle Nested Inputs/Outputs**:

- Remove `mask` from `forward`, `get_value`, and `select_action`.
- After model forward, unbind and stack the output (since it's fixed-size post-pooling).

```C++
std::pair<torch::Tensor, torch::Tensor>
BaseA2CAgent::forward(const torch::Tensor &batched_observations) {
   std::lock_guard lock(*this->model_mutex);
   torch::Tensor x = batched_observations.to(this->device).to(torch::kFloat);

   torch::Tensor actor_out = this->actor->forward(x);
   if (actor_out.is_nested()) {
       std::vector<torch::Tensor> components = actor_out.unbind();
       actor_out = torch::stack(components, /*dim=*/0);  // [B, NR_PASSES]
   }

   torch::Tensor critic_out = this->critic->forward(x);
   if (critic_out.is_nested()) {
       std::vector<torch::Tensor> components = critic_out.unbind();
       critic_out = torch::stack(components, /*dim=*/0);  // [B]
   }

   return {actor_out, critic_out};
}

// Similarly update get_value, select_action (remove mask, handle nested if needed)
```

4. **Update Training Loop in `train_a2c`**:

- Remove mask usage (e.g., in `agent->select_action(batched_observations, instructions_mask)` →
  `agent->select_action(batched_observations)`).
- For bootstrapping: `episode_values[T] = agent->get_value(environments.get_batched_observations());` works as-is (
  outputs regular tensor).
- Test for NaN/Inf in probabilities remains the same.

5. **Testing and Edge Cases**:

- Verify with small `B=2`, varying `N_i` (e.g., one with `N=2`, another `N=10`).
- If `N_i < max_qubits` (kernel size), ensure no errors (padding handles it).
- Monitor performance: If GPU efficiency drops too much, fall back to zero-padding but multiply input by mask:
  `x = x * mask.unsqueeze(-1);` before model (ensures padded rows are zeroed, already the case).
- If nested support causes issues, process each environment in a loop (unbatched), but this defeats parallelism.

This change makes the agent process only valid instructions, improving representation accuracy without large padded
blocks.

\boxed{Use ragged nested tensors by creating a nested_tensor from a vector of unpadded [N_i, IRP] tensors in
get_batched_observations_with_padding. Remove the mask.

Conv1d supports nested tensors natively by applying component-wise. Update custom modules (e.g. TransposeContiguous,
HalfScalingLayer) to handle nested inputs via map_nested_tensor in forward, recursing on components.

After model forward, unbind nested outputs and stack into regular [B, ...] tensors since post-pooling shapes are fixed.

Remove mask from forward/get_value/select_action.}

### SUpport Vector Machines

SVMs, or Support Vector Machines, are a type of supervised machine learning algorithm primarily used for classification
tasks (separating data into categories) and regression (predicting continuous values). The core idea is to find an
optimal "hyperplane" (a boundary line in 2D, plane in 3D, or higher-dimensional equivalent) that best divides your data
points into classes while maximizing the "margin" (the distance between the hyperplane and the nearest data points from
each class, called support vectors). This makes SVMs robust against overfitting and effective in high-dimensional
spaces.

For non-linearly separable data, SVMs use a "kernel trick" to implicitly map the input features into a
higher-dimensional space where a linear hyperplane can separate the classes, without actually computing the
high-dimensional coordinates (which would be computationally expensive). Common kernels include linear, polynomial,
radial basis function (RBF), and sigmoid. SVMs are great for scenarios with clear margins of separation, like image
classification or text categorization, but they can be slower to train on very large datasets compared to neural
networks.

Regarding your specific idea in the context of the A2C agent setup (where agents are parameterized by `max_qubits`, and
the Instruction Representation Padding/Size—or IRP—includes a fixed vector of size `max_qubits` for encoding qubit
involvement in gates via a signed multi-hot scheme: +1 for targets, -1 for controls): You're right that the current
fixed-size encoding limits scalability to circuits larger than `max_qubits` (you'd need to retrain or resize the model).
Switching to passing qubit "operations" (i.e., involvement in a gate) as raw indices instead of the multi-hot vector
would reduce IRP (dropping the `max_qubits` term) and remove the hard limit on qubit count.

However, as you noted, feeding raw indices directly into a neural network (like the Conv1D-based actor/critic in your
A2C setup) is problematic because NNs treat numerical inputs as continuous/ordinal features. This implies unintended
relationships: qubit 50 would be "larger" than qubit 1 (e.g., 50 times larger in a linear sense), or closer to qubit 49
than to qubit 0, which doesn't reflect reality—qubits are just categorical labels (arbitrary IDs) without inherent order
or magnitude in this context.

SVMs aren't a good fit to "separate the index values" or fix this interpretation issue in the way you're describing.
SVMs are end-to-end models for prediction (classification/regression), not feature preprocessors or encoders. While the
kernel trick could theoretically handle non-linear transformations of raw index features (e.g., via RBF kernel to
emphasize separations between specific index values), it wouldn't inherently "correct" the ordinal
misinterpretation—it's still operating on the raw numbers as-is. You'd essentially be using an SVM as your entire
model (replacing the NN), which doesn't align with your A2C reinforcement learning framework (where you need
actor/critic networks for policy and value estimation). SVMs also don't scale well to the sequential, variable-length
instruction data in your `InstructionsTensor` (they're better for fixed-feature vectors), and integrating them into a
deep RL setup like A2C would be awkward and likely less effective than modern NN approaches.

A better way to handle variable/unlimited qubits without the fixed multi-hot vector (and without raw indices' pitfalls)
is to treat qubit IDs as categorical variables and use **embeddings** (like word embeddings in NLP). Here's how you
could adapt your setup:

- **Representation Change**: For each instruction, instead of a fixed `max_qubits`-sized vector, represent qubit
  involvement as a list of indices (e.g., for a gate: list of control indices with a flag for control vs. target, or
  separate lists). The full observation becomes a variable-length sequence of instructions, each with variable-length
  lists of qubit indices (plus gate one-hot and params).

- **Model Adaptation**: Switch to a model that handles variable-length inputs natively, like a Transformer (e.g., via
  `torch::nn::TransformerEncoder`). For each qubit index in an instruction:
    - Embed the index into a dense vector (e.g., `torch::nn::Embedding` layer with vocab size large enough for any
      reasonable qubit count, say 1024+; embedding dim 16-64).
    - This learns a meaningful, low-dim representation for each qubit ID without assuming order (similar to how token
      IDs are embedded in LLMs).
    - Aggregate per-instruction qubits (e.g., sum/mean the embeddings or use attention over them).
    - Concat with gate one-hot and params, then process the sequence of instructions with the Transformer/RNN/Conv (
      Conv1D could still work if you pad per-batch or use nested tensors as discussed before).

This keeps IRP smaller and qubit-agnostic (no `max_qubits` in input size), allows arbitrary qubit counts, and avoids
ordinal assumptions. If your qubits have graph-like connectivity (e.g., in quantum circuits), a Graph Neural Network (
GNN) could model qubits as nodes with indices embedded similarly.

In short: SVMs are cool for classification with margins but won't solve your encoding problem—go with embeddings for
scalable, correct qubit representations in the NN.

### Principal Component Analysis and Exploratory Factor Analysis

Principal Component Analysis (PCA) and Exploratory Factor Analysis (EFA) are both statistical techniques commonly used
for dimensionality reduction and data exploration in machine learning and statistics, but they serve slightly different
purposes and make different assumptions about the data. I'll break them down step by step, including their key concepts,
uses, and differences, before addressing your specific idea about applying them to qubit indices in the context of your
A2C agent's encoding problem (where the agent is limited by a `max_qubits` parameter, and the Instruction Representation
Padding/Size—or IRP—relies on a fixed multi-hot vector for qubit involvement).

### What is Principal Component Analysis (PCA)?

PCA is an unsupervised machine learning algorithm primarily focused on **dimensionality reduction** by transforming a
high-dimensional dataset into a lower-dimensional one while preserving as much of the original variance (spread of data)
as possible. It does this by identifying new, uncorrelated variables called **principal components** (PCs), which are
linear combinations of the original features. These components are ordered by how much variance they explain—the first
PC captures the most variance, the second the next most (orthogonal to the first), and so on.

#### Key Steps in PCA:

1. **Standardize the data**: Center the features (subtract mean) and scale them (divide by standard deviation) to ensure
   equal weighting.
2. **Compute covariance matrix**: This shows how features vary together.
3. **Eigen decomposition**: Find eigenvalues (amount of variance) and eigenvectors (directions of new axes) of the
   covariance matrix.
4. **Select components**: Keep the top k eigenvectors (principal components) based on eigenvalues, projecting the data
   onto them for reduction.
5. **Transform data**: The result is a new dataset with fewer dimensions.

#### Assumptions and Characteristics:

- Assumes data is continuous and linearly related (though kernel PCA extends it to non-linear cases).
- Focuses on **maximizing variance**; it doesn't care about underlying "meaning" or latent constructs—it's purely
  mathematical.
- No error term; it treats all variance as signal (including noise).
- Output: A set of orthogonal components that explain cumulative variance (e.g., first few might explain 95% of total
  variance).

#### Common Uses in Machine Learning:

- **Dimensionality reduction**: To speed up training models (e.g., before feeding into a neural network) or handle the "
  curse of dimensionality" in high-feature datasets.
- **Data compression**: E.g., reducing image sizes while retaining key information.
- **Visualization**: Project high-dim data to 2D/3D for plotting (e.g., t-SNE is related but non-linear).
- **Noise reduction**: By discarding low-variance components assumed to be noise.
- **Feature engineering**: Combine correlated features into fewer, more informative ones (e.g., in finance for stock
  data or genomics for gene expression).
- Examples: Facial recognition (eigenfaces), anomaly detection, or preprocessing for clustering/classification.

PCA is implemented in libraries like scikit-learn (Python) or torch (for your NN context), and it's computationally
efficient for large datasets.

### What is Exploratory Factor Analysis (EFA)?

EFA is a statistical method from multivariate statistics (often in psychometrics and social sciences) used to **uncover
latent (unobserved) factors** that explain the correlations among a set of observed variables. It's "exploratory"
because you don't pre-specify the factor structure—you let the data reveal it. The goal is to identify underlying
constructs (e.g., "intelligence" as a factor explaining test scores in IQ research) and reduce data while interpreting
relationships.

#### Key Steps in EFA:

1. **Assess suitability**: Check correlations (e.g., Bartlett's test) and sampling adequacy (KMO measure).
2. **Extract factors**: Use methods like principal axis factoring or maximum likelihood to estimate initial factors
   based on shared variance (communality).
3. **Determine number of factors**: Use scree plots, eigenvalues (>1 rule), or parallel analysis.
4. **Rotate factors**: Apply orthogonal (e.g., Varimax) or oblique (e.g., Promax) rotation to make factors more
   interpretable (maximize high loadings, minimize low ones).
5. **Interpret**: Examine factor loadings (correlations between variables and factors) and name factors based on
   patterns.

#### Assumptions and Characteristics:

- Assumes continuous, normally distributed data (though robust versions exist).
- Models data as: Observed variable = Common factors + Unique variance (error/noise) + Specific variance.
- Focuses on **shared variance** (correlations); ignores unique variance as error.
- Factors can be correlated (oblique) or uncorrelated (orthogonal), unlike PCA's always-orthogonal components.
- More interpretive and theory-driven than PCA; often followed by confirmatory factor analysis (CFA) to test hypotheses.

#### Common Uses in Statistics:

- **Scale development**: In surveys/questionnaires to group items into subscales (e.g., personality tests like Big
  Five).
- **Identifying latent constructs**: In psychology (e.g., factors underlying depression symptoms) or marketing (customer
  preferences).
- **Data reduction with meaning**: When you want interpretable factors, not just variance maximization.
- **Hypothesis generation**: Explore data before building models (e.g., in education research for student performance
  factors).
- Examples: Reducing survey responses from 50 items to 5 factors, or analyzing economic indicators for underlying
  trends.

EFA is available in tools like SPSS, R (factanal), or Python (factor_analyzer).

### Key Differences Between PCA and EFA

- **Objective**: PCA maximizes total variance for reduction/compression; EFA identifies latent factors explaining
  correlations (more about structure discovery).
- **What They Model**: PCA treats all variance as important (components = linear combos of all variables); EFA separates
  common (shared) variance from unique/error variance.
- **Interpretability**: EFA often more interpretable after rotation; PCA is more mechanical.
- **Number of Components/Factors**: PCA can extract as many as variables (but you select based on variance); EFA
  typically extracts fewer based on shared structure.
- **When to Use**: PCA for pure reduction/noise filtering in ML; EFA for theory-building in social/behavioral sciences.
- **Similarities**: Both reduce dimensions, handle multicollinearity, and can be precursors to regression/clustering.

In practice, people sometimes use them interchangeably for reduction, but that's not ideal—PCA is better for
variance-focused tasks, EFA for correlation-focused ones.

### Can PCA or EFA Be Used on Qubit Indices to "Separate" Values and Fix the Ordinal Interpretation Issue?

Your assumption is correct: **No, PCA and EFA are not suitable for this purpose, and they wouldn't meaningfully "
separate the index values" to make raw qubit indices safe as inputs to your neural network without implying unwanted
ordinal relationships** (e.g., qubit 50 being "larger" than qubit 1). Here's why, tied to your agent's setup (where
qubit operations are currently encoded as a signed multi-hot vector of fixed size `max_qubits`, limiting scalability and
inflating IRP):

- **Qubit Indices Are Categorical Labels, Not Correlated Continuous Variables**: PCA and EFA are designed for datasets
  with multiple continuous features where you want to reduce dimensions by capturing variance (PCA) or latent
  correlations (EFA). A single qubit index (e.g., 5 or 50) per involvement in a gate/instruction isn't a
  high-dimensional dataset—it's just an ID/label without inherent numerical meaning or variance to analyze. Applying
  PCA/EFA to raw indices alone would be like running them on arbitrary category numbers (e.g., assigning 1=cat, 2=dog),
  which doesn't create meaningful separations; it might just scale or shift them without addressing ordinality.

- **They Don't "Correct" Ordinality**: These methods assume input features are numeric and meaningful as-is. If you feed
  raw indices, PCA might project them into new components, but those would still inherit the ordinal bias (e.g., higher
  indices could dominate variance if unevenly distributed). EFA looks for correlations, but isolated indices don't
  correlate in a latent-factor way—they're independent labels. For categorical data, you'd first need to one-hot
  encode (which brings you back to your fixed-size problem), then apply variants like Multiple Correspondence Analysis (
  MCA, a PCA analog for categoricals) or Categorical PCA. But even then, it's for reducing correlated categoricals
  across many variables/samples, not "separating" single IDs to make them NN-friendly.

- **Application to Your Data Wouldn't Help**: If you collected a dataset of many circuits and applied PCA/EFA to the
  qubit involvement patterns (e.g., treating each qubit's usage across instructions as features), you might reduce IRP
  by finding low-dim representations of qubit interactions. But this is offline preprocessing—it doesn't dynamically
  handle arbitrary qubit counts during training/inference, and it still requires bounding qubits initially. Plus, in
  quantum contexts like yours, qubit IDs don't have latent "factors" like survey items do; they're just arbitrary
  assignments (e.g., no "underlying construct" linking qubit 1 and 50). Results from show NNs for qubit
  characterization, but they use embeddings or direct features, not PCA/EFA on IDs.

- **Better Alternatives for Your Qubit Encoding Problem**: To pass qubit operations as indices (reducing IRP, removing
  `max_qubits` limit), treat them as categorical and use **learnable embeddings** (e.g., `torch::nn::Embedding` with a
  large vocab size like 1024+ for any qubit count). For each instruction, embed the list of control/target indices into
  dense vectors (e.g., dim 32), aggregate them (sum/mean/attention), and concat with gate one-hot/params. This avoids
  ordinality (embeddings learn similarity based on co-occurrence, not numeric value), handles unlimited qubits, and
  keeps IRP small. If graph structure matters (qubit connectivity), consider GNNs. Raw indices directly into Conv1D
  would indeed fail as you said—stick to embeddings for scalability.

In summary, PCA and EFA are powerful for reduction and exploration but not for fixing categorical encoding issues like
yours. If you have a dataset of circuits, you could experiment with PCA on the full IRP (post-one-hot) to compress it,
but that's orthogonal to removing the qubit limit.


