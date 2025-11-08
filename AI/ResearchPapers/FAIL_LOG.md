* **Primary trigger:** CUDA assert in `TensorCompare.cu:112` — probability tensor contains `inf`/`nan` or a negative value.
* **Runtime symptom:** Repeated `CUDA error: device-side assert triggered` (episodes 3975–3979).
* **Where it surfaced:** During scalar access `item()` via `_local_scalar_dense_cuda` (ep. 3975) and during tensor transfers/conversions (`copy_`, `_to_copy`, `to()`, `to_device`) in later episodes.
* **Propagation:** Errors reported asynchronously; stack traces may not point to the original kernel.
* **Final outcome:** Unhandled `c10::AcceleratorError` → process aborted (core dumped).
* **Context hints:** Logs mention debugging aids (`CUDA_LAUNCH_BLOCKING=1`, `TORCH_USE_CUDA_DSA`) and NVIDIA doc for `cudaErrorAssert`.

: [> ] 3975/1000000, 1/256 (0%) (Rollout A | Reward: 0.000000 | Nr qubits: 21 | Nr gates: 674) /pytorch/aten/src/ATen/native/cuda/TensorCompare.cu:112: _assert_async_cuda_kernel: block: [0
,0,0], thread: [0,0,0] Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed.
Episode 3975: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x9c (0x7ff45762fbcc in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10.so)
frame #1: <unknown function> + 0x1ccb9 (0x7ff4563becb9 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10_cuda.so)
frame #2: <unknown function> + 0x1bb3e32 (0x7ff459253e32 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cuda.so)
frame #3: at::native::_local_scalar_dense_cuda(at::Tensor const&) + 0x6e (0x7ff4592542ce in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x3b2f584 (0x7ff45b1cf584 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cuda.so)
frame #5: <unknown function> + 0x3b2f675 (0x7ff45b1cf675 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cuda.so)
frame #6: at::_ops::_local_scalar_dense::redispatch(c10::DispatchKeySet, at::Tensor const&) + 0x83 (0x7ff4e7a414d3 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #7: <unknown function> + 0x59ca230 (0x7ff4ea4dc230 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #8: <unknown function> + 0x59ca2d8 (0x7ff4ea4dc2d8 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #9: at::_ops::_local_scalar_dense::call(at::Tensor const&) + 0x17c (0x7ff4e7b176cc in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #10: at::native::item(at::Tensor const&) + 0x94 (0x7ff4e7042cd4 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #11: <unknown function> + 0x3858945 (0x7ff4e836a945 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #12: at::_ops::item::call(at::Tensor const&) + 0x17c (0x7ff4e792808c in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #13: int at::Tensor::item<int>() const + 0x2b (0x7ff4e8a48c4b in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #14: ./ai_pass_selector() [0x47e8bb]
frame #15: ./ai_pass_selector() [0x457fc2]
frame #16: ./ai_pass_selector() [0x436e98]
frame #17: <unknown function> + 0x295d0 (0x7ff456a295d0 in /lib64/libc.so.6)
frame #18: __libc_start_main + 0x80 (0x7ff456a29680 in /lib64/libc.so.6)
frame #19: ./ai_pass_selector() [0x447625]

: [> ] 3976/1000000, 1/256 (0%) (Rollout A | Reward: 0.000000 | Nr qubits: 28 | Nr gates: 369) Episode 3976: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x9c (0x7ff45762fbcc in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10.so)
frame #1: <unknown function> + 0x1ccb9 (0x7ff4563becb9 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10_cuda.so)
frame #2: <unknown function> + 0x1c64638 (0x7ff459304638 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0x229026b (0x7ff4e6da226b in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #4: at::native::copy_(at::Tensor&, at::Tensor const&, bool) + 0x7a (0x7ff4e6da39ca in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #5: at::_ops::copy_::call(at::Tensor&, at::Tensor const&, bool) + 0x1ae (0x7ff4e7cafd1e in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #6: at::native::_to_copy(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1e75 (0x7ff4e70ff7a5 in /home/ge7
8zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #7: <unknown function> + 0x36ede8f (0x7ff4e81ffe8f in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #8: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x10
9 (0x7ff4e7697c29 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #9: <unknown function> + 0x33c91ca (0x7ff4e7edb1ca in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #10: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1
09 (0x7ff4e7697c29 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #11: <unknown function> + 0x580cba2 (0x7ff4ea31eba2 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #12: <unknown function> + 0x580d062 (0x7ff4ea31f062 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #13: at::_ops::_to_copy::call(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x236 (0x7ff4e773fa06 in /home
/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #14: at::native::to(at::Tensor const&, c10::Device, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) + 0xf7 (0x7ff4e70fc837 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #15: <unknown function> + 0x38586ed (0x7ff4e836a6ed in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #16: at::_ops::to_device::call(at::Tensor const&, c10::Device, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) + 0x209 (0x7ff4e7926929 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_c
pu.so)
frame #17: ./ai_pass_selector() [0x4b3b51]
frame #18: ./ai_pass_selector() [0x47e790]
frame #19: ./ai_pass_selector() [0x457fc2]
frame #20: ./ai_pass_selector() [0x436e98]
frame #21: <unknown function> + 0x295d0 (0x7ff456a295d0 in /lib64/libc.so.6)
frame #22: __libc_start_main + 0x80 (0x7ff456a29680 in /lib64/libc.so.6)
frame #23: ./ai_pass_selector() [0x447625]

: [> ] 3977/1000000, 1/256 (0%) (Rollout A | Reward: 0.000000 | Nr qubits: 28 | Nr gates: 277) Episode 3977: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x9c (0x7ff45762fbcc in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10.so)
frame #1: <unknown function> + 0x1ccb9 (0x7ff4563becb9 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10_cuda.so)
frame #2: <unknown function> + 0x1c64638 (0x7ff459304638 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0x229026b (0x7ff4e6da226b in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #4: at::native::copy_(at::Tensor&, at::Tensor const&, bool) + 0x7a (0x7ff4e6da39ca in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #5: at::_ops::copy_::call(at::Tensor&, at::Tensor const&, bool) + 0x1ae (0x7ff4e7cafd1e in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #6: at::native::_to_copy(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1e75 (0x7ff4e70ff7a5 in /home/ge7
8zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #7: <unknown function> + 0x36ede8f (0x7ff4e81ffe8f in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #8: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x10
9 (0x7ff4e7697c29 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #9: <unknown function> + 0x33c91ca (0x7ff4e7edb1ca in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #10: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1
09 (0x7ff4e7697c29 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #11: <unknown function> + 0x580cba2 (0x7ff4ea31eba2 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #12: <unknown function> + 0x580d062 (0x7ff4ea31f062 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #13: at::_ops::_to_copy::call(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x236 (0x7ff4e773fa06 in /home
/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #14: at::native::to(at::Tensor const&, c10::Device, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) + 0xf7 (0x7ff4e70fc837 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #15: <unknown function> + 0x38586ed (0x7ff4e836a6ed in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #16: at::_ops::to_device::call(at::Tensor const&, c10::Device, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) + 0x209 (0x7ff4e7926929 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_c
pu.so)
frame #17: ./ai_pass_selector() [0x4b3b51]
frame #18: ./ai_pass_selector() [0x47e790]
frame #19: ./ai_pass_selector() [0x457fc2]
frame #20: ./ai_pass_selector() [0x436e98]
frame #21: <unknown function> + 0x295d0 (0x7ff456a295d0 in /lib64/libc.so.6)
frame #22: __libc_start_main + 0x80 (0x7ff456a29680 in /lib64/libc.so.6)
frame #23: ./ai_pass_selector() [0x447625]

: [> ] 3978/1000000, 1/256 (0%) (Rollout A | Reward: 0.000000 | Nr qubits: 20 | Nr gates: 912) Episode 3978: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x9c (0x7ff45762fbcc in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10.so)
frame #1: <unknown function> + 0x1ccb9 (0x7ff4563becb9 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10_cuda.so)
frame #2: <unknown function> + 0x1c64638 (0x7ff459304638 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0x229026b (0x7ff4e6da226b in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #4: at::native::copy_(at::Tensor&, at::Tensor const&, bool) + 0x7a (0x7ff4e6da39ca in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #5: at::_ops::copy_::call(at::Tensor&, at::Tensor const&, bool) + 0x1ae (0x7ff4e7cafd1e in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #6: at::native::_to_copy(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1e75 (0x7ff4e70ff7a5 in /home/ge7
8zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #7: <unknown function> + 0x36ede8f (0x7ff4e81ffe8f in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #8: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x10
9 (0x7ff4e7697c29 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #9: <unknown function> + 0x33c91ca (0x7ff4e7edb1ca in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #10: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1
09 (0x7ff4e7697c29 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #11: <unknown function> + 0x580cba2 (0x7ff4ea31eba2 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #12: <unknown function> + 0x580d062 (0x7ff4ea31f062 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #13: at::_ops::_to_copy::call(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x236 (0x7ff4e773fa06 in /home
/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #14: at::native::to(at::Tensor const&, c10::Device, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) + 0xf7 (0x7ff4e70fc837 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #15: <unknown function> + 0x38586ed (0x7ff4e836a6ed in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #16: at::_ops::to_device::call(at::Tensor const&, c10::Device, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) + 0x209 (0x7ff4e7926929 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_c
pu.so)
frame #17: ./ai_pass_selector() [0x4b3b51]
frame #18: ./ai_pass_selector() [0x47e790]
frame #19: ./ai_pass_selector() [0x457fc2]
frame #20: ./ai_pass_selector() [0x436e98]
frame #21: <unknown function> + 0x295d0 (0x7ff456a295d0 in /lib64/libc.so.6)
frame #22: __libc_start_main + 0x80 (0x7ff456a29680 in /lib64/libc.so.6)
frame #23: ./ai_pass_selector() [0x447625]

: [> ] 3979/1000000, 1/256 (0%) (Rollout A | Reward: 0.000000 | Nr qubits: 28 | Nr gates: 831) Episode 3979: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x9c (0x7ff45762fbcc in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10.so)
frame #1: <unknown function> + 0x1ccb9 (0x7ff4563becb9 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10_cuda.so)
frame #2: <unknown function> + 0x1c64638 (0x7ff459304638 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0x229026b (0x7ff4e6da226b in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #4: at::native::copy_(at::Tensor&, at::Tensor const&, bool) + 0x7a (0x7ff4e6da39ca in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #5: at::_ops::copy_::call(at::Tensor&, at::Tensor const&, bool) + 0x1ae (0x7ff4e7cafd1e in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #6: at::native::_to_copy(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1e75 (0x7ff4e70ff7a5 in /home/ge7
8zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #7: <unknown function> + 0x36ede8f (0x7ff4e81ffe8f in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #8: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x10
9 (0x7ff4e7697c29 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #9: <unknown function> + 0x33c91ca (0x7ff4e7edb1ca in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #10: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1
09 (0x7ff4e7697c29 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #11: <unknown function> + 0x580cba2 (0x7ff4ea31eba2 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #12: <unknown function> + 0x580d062 (0x7ff4ea31f062 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #13: at::_ops::_to_copy::call(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x236 (0x7ff4e773fa06 in /home
/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #14: at::native::to(at::Tensor const&, c10::Device, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) + 0xf7 (0x7ff4e70fc837 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #15: <unknown function> + 0x38586ed (0x7ff4e836a6ed in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #16: at::_ops::to_device::call(at::Tensor const&, c10::Device, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) + 0x209 (0x7ff4e7926929 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_c
pu.so)
frame #17: ./ai_pass_selector() [0x4b3b51]
frame #18: ./ai_pass_selector() [0x47e790]
frame #19: ./ai_pass_selector() [0x457fc2]
frame #20: ./ai_pass_selector() [0x436e98]
frame #21: <unknown function> + 0x295d0 (0x7ff456a295d0 in /lib64/libc.so.6)
frame #22: __libc_start_main + 0x80 (0x7ff456a29680 in /lib64/libc.so.6)
frame #23: ./ai_pass_selector() [0x447625]

terminate called after throwing an instance of 'c10::AcceleratorError'
 what(): CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x9c (0x7ff45762fbcc in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10.so)
frame #1: <unknown function> + 0x1ccb9 (0x7ff4563becb9 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libc10_cuda.so)
frame #2: <unknown function> + 0x1c64638 (0x7ff459304638 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0x229026b (0x7ff4e6da226b in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #4: at::native::copy_(at::Tensor&, at::Tensor const&, bool) + 0x7a (0x7ff4e6da39ca in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #5: at::_ops::copy_::call(at::Tensor&, at::Tensor const&, bool) + 0x1ae (0x7ff4e7cafd1e in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #6: at::native::_to_copy(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1e75 (0x7ff4e70ff7a5 in /home/ge7
8zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #7: <unknown function> + 0x36ede8f (0x7ff4e81ffe8f in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #8: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x10
9 (0x7ff4e7697c29 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #9: <unknown function> + 0x33c91ca (0x7ff4e7edb1ca in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #10: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1
09 (0x7ff4e7697c29 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #11: <unknown function> + 0x580cba2 (0x7ff4ea31eba2 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #12: <unknown function> + 0x580d062 (0x7ff4ea31f062 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #13: at::_ops::_to_copy::call(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x236 (0x7ff4e773fa06 in /home
/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #14: at::native::to(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, bool, std::optional<c10::MemoryFormat>) + 0x130 (0x7ff4e70fc5d0 in /home/ge7
8zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #15: <unknown function> + 0x3858685 (0x7ff4e836a685 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #16: at::_ops::to_dtype_layout::call(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, bool, std::optional<c10::MemoryFormat>) + 0x24b (0x7ff4e792
63bb in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)

frame #17: torch::jit::getWriteableTensorData(at::Tensor const&, bool) + 0x507 (0x7ff4eb6532f7 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #18: torch::jit::ScriptModuleSerializer::writeArchive(c10::IValue const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::alloca
tor<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, bool, bool) + 0x539 (0x7ff4ebc1a109 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #19: torch::jit::ScriptModuleSerializer::serialize(torch::jit::Module const&, std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<ch
ar>, std::allocator<char> >, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<st
d::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > > const&, bool, bool) + 0x12b (0x7ff4ebc1d00b in /home/ge
78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #20: torch::jit::ExportModule(torch::jit::Module const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allo
cator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char,
std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<ch
ar> > > > > const&, bool, bool, bool) + 0x5ae (0x7ff4ebc1e49e in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/libtorch_cpu.so)
frame #21: torch::serialize::OutputArchive::save_to(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x70 (0x7ff4ebf36b20 in /home/ge78zic2/Projects/MQSS-Passes-Suite/AI/external/libtorch/lib/lib
torch_cpu.so)
frame #22: ./ai_pass_selector() [0x47b225]
frame #23: ./ai_pass_selector() [0x484645]
frame #24: ./ai_pass_selector() [0x47de87]
frame #25: ./ai_pass_selector() [0x457fc2]
frame #26: ./ai_pass_selector() [0x436e98]
frame #27: <unknown function> + 0x295d0 (0x7ff456a295d0 in /lib64/libc.so.6)
frame #28: __libc_start_main + 0x80 (0x7ff456a29680 in /lib64/libc.so.6)
frame #29: ./ai_pass_selector() [0x447625]

Aborted (core dumped)



