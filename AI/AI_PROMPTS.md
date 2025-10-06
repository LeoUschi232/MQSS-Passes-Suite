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

## Unlimited Qubits

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
