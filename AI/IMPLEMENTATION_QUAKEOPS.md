```tablegen
/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QUAKE_OPS
#define CUDAQ_OPTIMIZER_DIALECT_QUAKE_OPS

//===----------------------------------------------------------------------===//
// High-level CUDA-Q support
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "cudaq/Optimizer/Dialect/CC/CCTypes.td"
include "cudaq/Optimizer/Dialect/Common/Traits.td"
include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.td"
include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.td"
include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.td"

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class QuakeOp<string mnemonic, list<Trait> traits = []> :
    Op<QuakeDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Alloca, Dealloc: allocation of quantum references
//===----------------------------------------------------------------------===//

def quake_AllocaOp : QuakeOp<"alloca", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Allocates a reference or collection of references to wires.";
  let description = [{
    The `alloca` operation allocates either a single quantum reference or a
    vector of quantum references. The size of the vector may be provided either
    statically as part of the type or dynamically as an integer-like argument,
    `size`. The return value will be a quantum reference, type `!quake.ref`,
    or a vector of such, type `!quake.veq<N>`.

    All references are assumed to be initialized to the value `|0>` initially.
    See also the `null_wire` op.

    The `QuakeAddDeallocs` and `UnwindLowering` passes will insert deallocation
    ops for the scopes in which allocations appear automatically. This is
    helpful for generating code for targets such as QIR, which require
    allocation/deallocation pairs.

    Examples:
    ```mlir
      // Allocate a single qubit
      %qubit = quake.alloca !quake.ref

      // Allocate a qubit register with a size known at compilation time
      %veq = quake.alloca !quake.veq<4> {name = "quantum"}

      // Allocate a qubit register with a size known at runtime time
      %veq = quake.alloca(%size : i32) !quake.veq<?>
    ```

    Canonicalization for this op will fold constant vector sizes directly into
    the type.

    See DeallocOp.
  }];

  let arguments = (ins
    Optional<AnySignlessInteger>:$size
  );
  let results = (outs
    AnyRefType:$ref_or_vec
  );

  let builders = [
    OpBuilder<(ins ), [{
      return build($_builder, $_state, $_builder.getType<RefType>(), {});
    }]>,
    OpBuilder<(ins "size_t":$size), [{
      return build($_builder, $_state, $_builder.getType<VeqType>(size), {});
    }]>,
    OpBuilder<(ins "mlir::Type":$ty), [{
      return build($_builder, $_state, ty, {});
    }]>
  ];

  let assemblyFormat = [{
    qualified(type($ref_or_vec)) (`[` $size^ `:` type($size) `]`)? attr-dict
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool hasInitializedState() {
      auto *self = getOperation();
      return self->hasOneUse() &&
        mlir::isa<quake::InitializeStateOp>(*self->getUsers().begin());
    }

    quake::InitializeStateOp getInitializedState();
  }];
}

def quake_InitializeStateOp : QuakeOp<"init_state",
    [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Initialize the quantum state to a specific complex vector.";
  let description = [{
    Given a !cc.ptr pointing to a complex data array of size 2**N, where N is
    the number of qubits in the targets operand, initialize the state of those
    target qubits to the provided state vector. This operation returns a new
    quake.veq instance. There should be no other uses of the input veq value,
    \em{targets}, that was allocated. This supports a RAII (resource allocation
    is initialization) semantics on the qubits in the vector.
  }];

  let arguments = (ins
    VeqType:$targets,
    AnyStateInitType:$state
  );
  let results = (outs VeqType);
  
  let assemblyFormat = [{
    $targets `,` $state `:` functional-type(operands, results) attr-dict
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def quake_DeallocOp : QuakeOp<"dealloc"> {
  let summary = "Deallocates a collection of qubits.";
  let description = [{
    The `dealloc` operation deallocates a quantum reference. The deallocation
    can be a single quantum reference, `!quake.ref`, or a vector of quantum
    references, `!quake.veq<N>`.

    Deallocations are automatically inserted by the `AddDeallocs` and
    `UnwindLowering` passes.

    Example:
    ```mlir
      %1 = quake.alloca !quake.veq<4>
      ...
      quake.dealloc %1 : !quake.veq<4>
    ```

    See AllocaOp.
  }];

  let arguments = (ins
    Arg<AnyRefType, "qubit reference (or vector) to deallocate",
    [MemFree]>:$reference
  );

  let assemblyFormat = [{
    $reference `:` qualified(type($reference)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Veq reference manipulation primitives
//===----------------------------------------------------------------------===//

def quake_ConcatOp : QuakeOp<"concat", [Pure]> {
  let summary = "Construct a veq from a list of other ref/veq values.";
  let description = [{
    The `concat` operation allows one to concatenate a list of SSA-values of
    either type Ref or Veq into a new Veq vector.

    Example:
    ```mlir
      %veq = quake.concat %r1, %v1, %r2 : (!quake.ref, !quake.veq<?>,
                                           !quake.ref) -> !quake.veq<?>
    ```
  }];

  let arguments = (ins Variadic<AnyRefType>:$qbits);
  let results = (outs VeqType);

  let assemblyFormat = [{
    $qbits attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

def quake_ExtractRefOp : QuakeOp<"extract_ref", [Pure]> {
  let summary = "Extract a quantum reference from a quantum vector.";
  let description = [{
    The `extract_ref` operation extracts a quantum reference from a vector of
    quantum references.

    The following example extracts the quantum reference at position 0 from a
    vector of references. The vector, in this case, has unknown size.

    Example:
    ```mlir
      %zero = arith.constant 0 : i32
      %qr = quake.extract_ref %qv[%zero] : (!quake.veq<?>, i32) -> !quake.ref
    ```
  }];

  let arguments = (ins
    VeqType:$veq,
    Optional<AnySignlessIntegerOrIndex>:$index,
    I64Attr:$rawIndex
  );
  let results = (outs RefType:$ref);

  let builders = [
    OpBuilder<(ins "mlir::Value":$veq, "mlir::Value":$index,
                   "mlir::IntegerAttr":$rawIndex), [{
      return build($_builder, $_state, $_builder.getType<RefType>(), veq,
                   index, rawIndex);
    }]>,
    OpBuilder<(ins "mlir::Value":$veq, "mlir::Value":$index), [{
      return build($_builder, $_state, $_builder.getType<RefType>(), veq,
                   index, ExtractRefOp::kDynamicIndex);
    }]>,
    OpBuilder<(ins "mlir::Value":$veq, "std::size_t":$rawIndex), [{
      auto i64Ty = $_builder.getI64Type();
      return build($_builder, $_state, $_builder.getType<RefType>(), veq,
                   mlir::Value{}, mlir::IntegerAttr::get(i64Ty, rawIndex));
    }]>
  ];

  let assemblyFormat = [{
    $veq `[` custom<RawIndex>($index, $rawIndex) `]` `:`
      functional-type(operands, results) attr-dict
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static constexpr std::size_t kDynamicIndex =
      std::numeric_limits<std::size_t>::max();

    bool hasConstantIndex() { return !getIndex(); }
    std::size_t getConstantIndex() { return getRawIndex(); }
  }];
}

def quake_RelaxSizeOp : QuakeOp<"relax_size", [Pure]> {
  let summary = "Relax the constant size on a !veq to be unknown.";
  let description = [{
    At times, the IR needs to forget the length of an SSA-value of type
    `!quake.veq<N>` and demote it to type `!quake.veq<?>` where the size is
    said to be unknown. This demotion is required to preserve a valid,
    strongly-typed IR.

    Example:
    ```mlir
      %uqv = quake.relax_size %qv : (!quake.veq<4>) -> !quake.veq<?>
    ```
  }];

  let arguments = (ins VeqType:$inputVec);
  let results = (outs VeqType);

  let assemblyFormat = [{
    $inputVec `:` functional-type(operands, results) attr-dict
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def quake_SubVeqOp : QuakeOp<"subveq", [AttrSizedOperandSegments, Pure]> {
  let summary = "Extract a subvector from a veq reference value.";
  let description = [{
    The `subveq` operation returns a subvector of references, type
    `!quake.veq<N>` from a vector of references, type `!quake.veq<M>`, where
    `M >= N`.

    In the following example, the operation produces an SSA-value with 5
    references. These references may be indexed from 0 to 4 via `%qr` and are
    the same references as those from 2 to 6 indexed via `%qv`. Specifically,
    the returned vector, `%qr`, is not a constructor and does not own the
    references; it simply makes a copy as a new (shorter) vector. Therefore,
    subvectors need never be deallocated.

    Example:
    ```mlir
      %0 = arith.constant 2 : i32
      %1 = arith.constant 6 : i32
      %qr = quake.subveq %qv, %0, %1 : (!quake.veq<?>, i32, i32) ->
                                        !quake.veq<5>
    ```
  }];

  let arguments = (ins
    VeqType:$veq,
    Optional<AnySignlessIntegerOrIndex>:$lower,
    Optional<AnySignlessIntegerOrIndex>:$upper,
    I64Attr:$rawLower,
    I64Attr:$rawUpper
  );
  let results = (outs VeqType:$qsub);

  let assemblyFormat = [{
    $veq `,` custom<RawIndex>($lower, $rawLower) `,` custom<RawIndex>($upper,
      $rawUpper) `:` functional-type(operands, results) attr-dict
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::Type":$veqTy, "mlir::Value":$input,
                   "mlir::Value":$lower, "mlir::Value":$upper), [{
      return build($_builder, $_state, veqTy, input, lower, upper,
        quake::SubVeqOp::kDynamicIndex, quake::SubVeqOp::kDynamicIndex);
    }]>,
    OpBuilder<(ins "mlir::Type":$veqTy, "mlir::Value":$input,
                   "std::int64_t":$lower, "std::int64_t":$upper), [{
      return build($_builder, $_state, veqTy, input, {}, {}, lower, upper);
    }]>
  ];
  
  let extraClassDeclaration = [{
    static constexpr std::size_t kDynamicIndex =
      std::numeric_limits<std::size_t>::max();

    bool hasConstantLowerBound() { return getRawLower() != kDynamicIndex; }
    bool hasConstantUpperBound() { return getRawUpper() != kDynamicIndex; }
    std::size_t getConstantLowerBound() { return getRawLower(); }
    std::size_t getConstantUpperBound() { return getRawUpper(); }
  }];
}

def quake_VeqSizeOp : QuakeOp<"veq_size", [Pure]> {
  let summary = "Return the size of a veq.";
  let description = [{
    Returns the size of a value of type `!quake.veq<n>`. If the vector has a
    static size, the static size is returned (effectively as a constant). If
    the size of the vector is dynamic, the size value will be an SSA-value.

    Examples:
    ```mlir
      %0 = quake.alloca !quake.veq<4>
      // %1 will be 4 with canonicalization.
      %1 = quake.veq_size %0 : (!quake.veq<4>) -> i64
      
      %2 = ... : !quake.veq<?>
      // %3 may not be computed until runtime.
      %3 = quake.veq_size %2 : (!quake.veq<?>) -> i64
    ```
  }];

  let arguments = (ins VeqType:$veq);
  let results = (outs AnySignlessIntegerOrIndex:$size);

  let assemblyFormat = [{
    $veq `:` functional-type(operands, results) attr-dict
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Application, ComputeAction(Uncompute)
//===----------------------------------------------------------------------===//

def quake_ApplyOp : QuakeOp<"apply",
    [AttrSizedOperandSegments, CallOpInterface]> {
  let summary = "Abstract application of a function in Quake.";
  let description = [{
    User-defined kernels define both predicated and unpredicated functions.
    The predicated form is implicitly defined. To simplify lowering, the
    unpredicated function may be defined while an ApplyOp may use the
    implied predicated function. A subsequent pass will then instantiate both
    the unpredicated and predicated variants.
  }];

  let arguments = (ins
    OptionalAttr<SymbolRefAttr>:$callee,
    Variadic<cc_CallableType>:$indirect_callee, // must be 0 or 1 element
    UnitAttr:$is_adj,
    Variadic<AnyQType>:$controls,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>);

  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$retTy,
                   "mlir::SymbolRefAttr":$callee,
                   "mlir::UnitAttr":$is_adj,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$args), [{
      return build($_builder, $_state, retTy, callee, mlir::ValueRange{},
                   is_adj, controls, args);
    }]>,
    OpBuilder<(ins "mlir::TypeRange":$retTy,
                   "mlir::SymbolRefAttr":$callee,
                   "bool":$is_adj,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$args), [{
      return build($_builder, $_state, retTy, callee, mlir::ValueRange{},
                   is_adj, controls, args);
    }]>,
    OpBuilder<(ins "mlir::TypeRange":$retTy,
                   "mlir::Value":$callable,
                   "mlir::UnitAttr":$is_adj,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$args), [{
      return build($_builder, $_state, retTy, mlir::SymbolRefAttr{},
                   mlir::ValueRange{callable}, is_adj, controls, args);
    }]>,
    OpBuilder<(ins "mlir::TypeRange":$retTy,
                   "mlir::Value":$callable,
                   "bool":$is_adj,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$args), [{
      return build($_builder, $_state, retTy, mlir::SymbolRefAttr{},
                   mlir::ValueRange{callable}, is_adj, controls, args);
    }]>
  ];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getCalleeAttrNameStr() { return "callee"; }

    mlir::FunctionType getFunctionType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      if (getControls().empty())
        return {operand_begin(), operand_end()};
      return {getArgs().begin(), getArgs().end()};
    }

    bool applyToVariant() {
      return getIsAdj() || !getControls().empty();
    }

    /// Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>(getCalleeAttrName());
    }
  }];
}

// A ComputeActionOp will be transformed into a series of CallOps.
def quake_ComputeActionOp : QuakeOp<"compute_action"> {
  let summary = "Captures the compute/action/uncompute high-level idiom.";
  let description = [{
    CUDA-Q supports the high-level compute, action, uncompute idiom by
    providing a custom template function (class) that takes pure kernels (a
    callable like a λ) as arguments. This operation captures uses of the idiom
    and can be systematically expanded into a quantum circuit via successive
    transformations.

    The `is_dagger` attribute can be used to "reverse" this idiom to one of
    uncompute, action, compute.

    The uncompute step is generated automatically by generating the adjoint of
    the compute kernel.
  }];

  let arguments = (ins
    UnitAttr:$is_dagger,
    cc_CallableType:$compute,
    cc_CallableType:$action
  );

  let assemblyFormat = [{
    (`<` `dag` $is_dagger^ `>`)? $compute `,` $action `:`
      qualified(type(operands)) attr-dict
  }];
}

def quake_ApplyNoiseOp : QuakeOp<"apply_noise", [AttrSizedOperandSegments]> {
  let summary = "Apply a noise operation to qubits.";
  let description = [{
    This operation provides support for the `cudaq::apply_noise` template
    function. This function is only valid is simulation contexts where the
    simulator is part of the same process as the C++ host executable itself.

    A noise operator is the application of a Kraus channel to a selected set
    of qubits. This is a point-wise annotation approach that a user might
    deploy to introduce "noise" to their circuit under simulation. It is unlike
    a general (unitary) gate application in that there is no notion of controls
    or an adjoint.
  }];

  let arguments = (ins
    OptionalAttr<FlatSymbolRefAttr>:$noise_func,
    Optional<AnySignlessInteger>:$key,
    Variadic<AnyType>:$parameters,
    Variadic<NonStruqRefType>:$qubits
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "mlir::StringRef":$noise_func,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, mlir::TypeRange{},
        mlir::FlatSymbolRefAttr::get($_builder.getContext(), noise_func), {},
        parameters, targets);
    }]>,
    OpBuilder<(ins "mlir::FlatSymbolRefAttr":$noise_func,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, mlir::TypeRange{}, noise_func, {},
        parameters, targets);
    }]>,
    OpBuilder<(ins "mlir::Value":$key,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, mlir::TypeRange{},
        mlir::FlatSymbolRefAttr{}, key, parameters, targets);
    }]>
  ];

  let extraClassDeclaration = [{
    static constexpr mlir::StringRef getNoiseFuncAttrNameStr() {
      return "noise_func";
    }
  }];
}

//===----------------------------------------------------------------------===//
// Memory and register conversion instructions: These operations are useful for
// intermediate conversions between memory-SSA and value-SSA semantics and vice
// versa of the IR. They mainly exist during the conversion process.
//===----------------------------------------------------------------------===//

def quake_UnwrapOp : QuakeOp<"unwrap"> {
  let summary = "Unwrap a reference to a wire and return the wire value.";
  let description = [{
    A quantum reference is an SSA-value that is associated with a volatile
    quantum wire. The unwrap operation allows conversion from the reference
    value semantics (memory SSA) to the volatile quantum wire value semantics
    when/as desired. The binding of a reference value corresponds to a
    particular data flow of volatile quantum wire values.

    Unwrap and wrap operations should (typically) form pairs as in the following
    example.

    ```mlir
      %0 = ... : !quake.ref
      %1 = quake.unwrap %0 : (!quake.ref) -> !quake.wire
      %2 = quake.rx (%dbl) %1 : (f64, !quake.wire) -> !quake.wire
      quake.wrap %2 to %0 : !quake.wire, !quake.ref
    ```
  }];

  let arguments = (ins Arg<RefType,"",[MemRead]>:$ref_value);
  let results = (outs WireType);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $ref_value `:` functional-type(operands, results) attr-dict
  }];
}

def quake_WrapOp : QuakeOp<"wrap"> {
  let summary = "Wrap a wire value with its reference.";
  let description = [{
    Each data flow of a volatile wire in a quantum circuit may be associated
    and identified with an invariant and unique quantum reference value of type
    `!ref`. The wrap operation provides a means of returning a quantum value,
    a wire, back to the reference value domain.
  }];

  let arguments = (ins
    WireType:$wire_value,
    Arg<RefType,"",[MemWrite]>:$ref_value
  );
  let assemblyFormat = [{
    $wire_value `to` $ref_value `:` qualified(type(operands)) attr-dict
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Qubit value semantics: NullWire, Sink
//===----------------------------------------------------------------------===//

def quake_NullWireOp : QuakeOp<"null_wire"> {
  let summary = "Initial state of a wire.";
  let description = [{
    |0> - the initial state of a wire when first constructed. A wire is assumed
    to be defined in the |0> state as its initial state.

    There is an unlimited number of virtual null wires. See `quake.borrow_wire`
    when constraining the number of qubits to a finite set.
  }];

  // This op has no dependence on classical memory and should not need to be
  // ordered using memory constraints. They are used as a workaround to prevent
  // MLIR from mistakenly reordering operations when the IR is a mix of quantum
  // value and quantum reference semantics. This workaround is applied to sink,
  // borrow_wire, and return_wire as well.
  let results = (outs
    Arg<WireType, "wire created", [MemRead, MemWrite]>
  );
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";
}

def quake_SinkOp : QuakeOp<"sink"> {
  let summary = "Sink for a qubit that will no longer be used in the circuit.";
  let description = [{
    The `quake.sink` operation is used to mark a particular wire in the value
    semantics as "free" at the end of a circuit. `quake.sink` is specifically
    used to free (virtual) wires obtained via `quake.null_wire`.

    This op is similar to the dealloc op in the reference semantics. It is also
    similar to `quake.return_wire` when using wire sets.

    Example:
    ```mlir
      quake.sink %0 : !quake.wire
    ```
  }];

  let arguments = (ins
    Arg<WireType, "wire to sink", [MemRead, MemWrite]>:$target
  );
  let assemblyFormat = [{
     $target `:` qualified(type(operands)) attr-dict
  }];
}

def quake_NullCableOp : QuakeOp<"null_cable"> {
  let summary = "Create a new cable of individual wires, all at null state.";
  let description = [{
    This operation can be used as syntactic sugar to create a number of wires,
    each of which is in the null state, all in one operation. This op has
    identical semantics to a series of `quake.null_wire` operations.
  }];

  let results = (outs 
    Arg<CableType, "cable created", [MemRead, MemWrite]>:$cableType
  );

  let assemblyFormat = [{
    type($cableType) attr-dict
  }];
}

def quake_BundleCableOp : QuakeOp<"bundle_cable"> {
  let summary = "Bundle a list of wire values into a single cable value.";
  let description = [{
    This operation can be used as syntactic sugar to logically bind any number
    of quantum wires together in a cable value. The resulting cable must have
    a size that exactly matches the arity of the arguments.

    The order of the wires as presented at the `bundle_cable` operation will be
    the exact same order as will be found in the results of a `terminate_cable`
    operation.
  }];

  let arguments = (ins Variadic<WireType>:$wires);
  let results = (outs CableType);

  let assemblyFormat = [{
    $wires `:` functional-type(operands, results) attr-dict
  }];
  let hasVerifier = 1;
}

def quake_TerminateCableOp : QuakeOp<"terminate_cable"> {
  let summary = "Termination of a cable of wires, exposing all the wires.";
  let description = [{
    This operation can be used as syntactic sugar at the termination of a cable
    of wires. This exposed all the wires in a cable as individual wire values.
    All wires are linear types and must be used exactly one time.
  }];
  
  let arguments = (ins CableType:$cable);
  let results = (outs Variadic<WireType>);

  let assemblyFormat = [{
    $cable `:` functional-type(operands, results) attr-dict
  }];
  let hasVerifier = 1;
}

def quake_SinkCableOp : QuakeOp<"sink_cable"> {
  let summary = "Sink a cable of quantum wires.";
  let description = [{
    This operation can be used as syntactic sugar for connecting a set of wire
    values to a series of `quake.sink` operations. It can be used, possibly, to
    declutter code by combining a long sequence of `quake.sink` operations.
  }];

  let arguments = (ins
    Arg<CableType, "cable destroyed", [MemRead, MemWrite]>:$cable
  );

  let assemblyFormat = [{
    $cable `:` type($cable) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Qubit assignment: WireSet, BorrowWire, ReturnWire
//===----------------------------------------------------------------------===//

def quake_WireSetOp : QuakeOp<"wire_set", [IsolatedFromAbove, Symbol]> {
  let summary = "Define a set of wires with a constant cardinality.";
  let description = [{
    At some point during a compilation, we may wish to refine our quantum
    circuits from using an unlimited set of virtual references/wires to using
    a finite set of qubits/wires.

    A wire set is a top-level object in the module that defines the properties
    of the target that we wish to reason about.

    Example:
    ```mlir
      quake.wire_set @phys[8]
    ```
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    I32Attr:$cardinality,
    OptionalAttr<ElementsAttr>:$adjacency
  );

  let hasCustomAssemblyFormat = 1;
}

def quake_BorrowWireOp : QuakeOp<"borrow_wire"> {
  let summary = "Borrow a specific wire from a wire set.";
  let description = [{
    To obtain a specific wire from a wire set, the wire must be borrowed. Once
    it is borrowed, it is undefined for any other operation to attempt to
    borrow the same wire (as determined by the identity value).

    It is an error to specify an identity that is outside the interval
    `[0 .. n)` where `n` is the cardinality of the wire set. (This will raise
    a verification error.)

    A borrowed wire must be returned to the wire set when the circuit is no
    longer using it. See `return_wire`.

    Example:
    ```mlir
      quake.wire_set @phys[8]

      func.func @qernel() {
        ...
        %6 = quake.borrow_wire @phys[4] : !wire
        ...
        quake.return_wire %6 : !wire
        ...
      }
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$set_name,
    I32Attr:$identity
  );
  let results = (outs
    Arg<WireType, "wire borrowed", [MemRead, MemWrite]>
  );
  
  let hasVerifier = 1;
  let assemblyFormat = [{
    $set_name `[` $identity `]` `:` type(results) attr-dict
  }];
}

def quake_ReturnWireOp : QuakeOp<"return_wire"> {
  let summary = "Return a borrowed wire to a wire set.";
  let description = [{
    When a wire is no longer needed for further use it must be returned to the
    wire set from which it was borrowed. The `return_wire` operation returns
    the wire.
  }];

  let arguments = (ins
    Arg<WireType, "wire to return", [MemRead, MemWrite]>:$target
  );
  let assemblyFormat = "$target `:` type(operands) attr-dict";
}

//===----------------------------------------------------------------------===//
// Struq handling
//===----------------------------------------------------------------------===//

def quake_MakeStruqOp : QuakeOp<"make_struq", [Pure]> {
  let summary = "create a quantum struct from a set of quantum references";
  let description = [{
    Given a list of values of quantum reference type, creates a new quantum
    product reference type. This is a logical grouping and does not imply any
    new quantum references are created.

    This operation can be useful for grouping a number of values of type `veq`
    into a logical product type, which may be passed to a pure device kernel
    as a single unit, for example. These product types may always be erased into
    a vector of the quantum references used to compose them via a make_struq op.
  }];

  let arguments = (ins Variadic<NonStruqRefType>:$veqs);
  let results = (outs StruqType);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $veqs `:` functional-type(operands, results) attr-dict
  }];
}

def quake_GetMemberOp : QuakeOp<"get_member", [Pure]> {
  let summary = "extract quantum references from a quantum struct";
  let description = [{
    The get_member operation can be used to extract a set of quantum references
    from a quantum struct (product) type. The fields in the quantum struct are
    indexed from 0 to $n-1$ where $n$ is the number of fields. An index outside
    of this range will produce a verification error.
  }];

  let arguments = (ins
    StruqType:$struq,
    I32Attr:$index
  );
  let results = (outs NonStruqRefType);
  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let assemblyFormat = [{
    $struq `[` $index `]` `:` functional-type(operands, results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ToControl, FromControl pair
//===----------------------------------------------------------------------===//

def quake_ToControlOp : QuakeOp<"to_ctrl", [Pure]> {
  let summary = "Convert a wire value to a control value.";
  let description = [{
    This operation makes the conversion of a wire value to a control value
    explicit in the quake IR. These values have different semantics in the IR.
    This op ensures these semantics via the type system.

    A value of type control is (nearly) an SSA-value. Once defined, via the
    `to_ctrl` operation, it can be used as an argument to other operations.
    These uses are qualified. They must be in control argument positions and
    these operations must dominate a `from_ctrl` operation that returns the
    control qubit back to a wire. The operand value and result value of a
    `to_ctrl` may NOT be used as arguments to the same operation.
  }];

  let arguments = (ins WireType:$qubit);
  let results = (outs ControlType);

  let assemblyFormat = [{
    $qubit `:` functional-type(operands, results) attr-dict
  }];
}

def quake_FromControlOp : QuakeOp<"from_ctrl", [Pure]> {
  let summary = "Convert a control value to a wire value.";
  let description = [{
    This operation makes the conversion of a control value to a wire value
    explicit in the quake IR. These values have different semantics in the IR.
    This op ensures these semantics via the type system.
  }];

  let arguments = (ins ControlType:$ctrlbit);
  let results = (outs WireType);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $ctrlbit `:` functional-type(operands, results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Reset
//===----------------------------------------------------------------------===//

def quake_ResetOp : QuakeOp<"reset", [QuantumGate,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Reset the wire to the |0> (|0..0>) state.";
  let description = [{
    The `quake.reset` operation resets a wire to the |0> (|0..0>) state. It
    may take an argument that is either a reference to a wire, type `!ref`,
    or a wire, type `!wire`.

    Example:
    ```mlir
      quake.reset %0 : (!quake.ref) -> ()
      %2 = quake.reset %1 : (!quake.wire) -> !quake.wire
    ```
  }];

  let arguments = (ins
    AnyQTargetType:$targets
  );
  let results = (outs
    Variadic<WireType>:$wires
  );
  let hasVerifier = 1;

  let assemblyFormat = [{
     $targets `:` functional-type(operands, results) attr-dict
  }];

  let extraClassDeclaration = [{
    void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::
        EffectInstance<mlir::MemoryEffects::Effect>> &effects) {
      quake::getResetEffectsImpl(effects, getTargets());
    }
  }];
}

//===----------------------------------------------------------------------===//
// Measurements, Discriminate
//===----------------------------------------------------------------------===//

class Measurement<string mnemonic> : QuakeOp<mnemonic, [MeasurementInterface,
    QuantumMeasure, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins
    Variadic<AnyQTargetType>:$targets,
    OptionalAttr<StrAttr>:$registerName
  );
  let results = (outs
    AnyTypeOf<[MeasureType, StdvecOf<[MeasureType]>]>:$measOut,
    Variadic<WireType>:$wires
  );

  let assemblyFormat = [{
    $targets (`name` $registerName^)? `:` functional-type(operands, results)
      attr-dict
  }];

  code OpBaseDeclaration = [{
    void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
      mlir::MemoryEffects::Effect>> &effects) {
      quake::getMeasurementEffectsImpl(effects, getTargets());
    }
  }];

  let hasVerifier = 1;
}

def MxOp : Measurement<"mx"> {
  let summary = "Measurement along the x-axis";
  let description = [{
    The `mx` operation measures the state of qubits into classical bits
    represented by a `i1` (or a vector of `i1`), along the x-axis.

    The state of the qubits is collapsed into one of the computational basis
    states, i.e., either |0> or |1>. A `reset` operation can guarantee that the
    qubit returns to a |0> state, and thus it can be used for further
    computation. Another option is to deallocate the qubit using `dealloc`.
  }];
  let extraClassDeclaration = OpBaseDeclaration;
}

def MyOp : Measurement<"my"> {
  let summary = "Measurement along the y-axis";
  let description = [{
    The `my` operation measures the state of qubits into classical bits
    represented by a `i1` (or a vector of `i1`), along the y-axis.

    The state of the qubit is collapsed into one of the computational basis
    states, i.e., either |0> or |1>. A `reset` operation can guarantee that the
    qubit returns to a |0> state, and thus it can be used for further
    computation. Another option is to deallocate the qubit using `dealloc`.
  }];
  let extraClassDeclaration = OpBaseDeclaration;
}

def MzOp : Measurement<"mz"> {
  let summary = "Measurement along the z-axis";
  let description = [{
    The `mz` operation measures the state of qubits into a classical bits
    represented by a `i1` (or a vector of `i1`), along the z-axis---the
    so-called computational basis.

    The state of the qubit is collapsed into one of the computational basis
    states, i.e., either |0> or |1>. A `reset` operation can guarantee that the
    qubit returns to a |0> state, and thus it can be used for further
    computation. Another option is to deallocate the qubit using `dealloc`.
  }];
  let extraClassDeclaration = OpBaseDeclaration;
}

def quake_DiscriminateOp : QuakeOp<"discriminate", [Pure]> {
  let summary = "Converts a measurement to a classical integral value.";
  let description = [{
    Quake's measurement operators return a value of type `!quake.measure`. The
    discriminate operation converts a value of type measure to a classical
    integral value. This value is typically an `i1` type, but might be `i2` for
    qutrits, or even an `i8` for general qudits.

    While a measurement of a wire changes/corrupts the state of the wire, the
    model maintains that a `!quake.measure` value is non-volatile. Therefore,
    multiple applications of discriminate on the same `!quake.measure` value
    will yield the same result value for a given result type.
  }];

  let arguments = (ins
    AnyTypeOf<[MeasureType, StdvecOf<[MeasureType]>]>:$measurement
  );
  let results = (outs
    AnyTypeOf<[AnySignlessInteger, StdvecOf<[AnySignlessInteger]>]>
  );

  let assemblyFormat = [{
    $measurement `:` functional-type(operands, results) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Quantum gates
//===----------------------------------------------------------------------===//

class QuakeOperator<string mnemonic, list<Trait> traits = [],
                    dag extraArgs = (ins)>
    : QuakeOp<mnemonic,
        !listconcat([QuantumGate, AttrSizedOperandSegments, OperatorInterface,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>], traits)> {

  let arguments = !con(extraArgs, (ins
    UnitAttr:$is_adj,
    Variadic<AnyFloat>:$parameters,
    Variadic<AnyQType>:$controls,
    Variadic<AnyQTargetType>:$targets,
    OptionalAttr<DenseBoolArrayAttr>:$negated_qubit_controls
  ));
  let results = (outs
    Variadic<WireType>:$wires
  );

  let builders = [
    OpBuilder<(ins "mlir::UnitAttr":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::DenseBoolArrayAttr":$negates), [{
      return build($_builder, $_state, mlir::TypeRange{}, is_adj, parameters,
                   controls, targets, negates);
    }]>,
    OpBuilder<(ins "bool":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::DenseBoolArrayAttr":$negates), [{
      return build($_builder, $_state, mlir::TypeRange{}, is_adj, parameters,
                   controls, targets, negates);
    }]>,
    OpBuilder<(ins "bool":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, is_adj, parameters, controls, targets,
                   {});
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, /*is_adj=*/false, parameters, controls,
                   targets);
    }]>,
    OpBuilder<(ins "bool":$is_adj,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, is_adj, mlir::ValueRange{}, controls,
                   targets);
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, /*is_adj=*/false, controls, targets);
    }]>,
    OpBuilder<(ins "bool":$is_adj,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, is_adj, mlir::ValueRange{}, targets);
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, /*is_adj=*/false, targets);
    }]>
  ];

  let assemblyFormat = [{
    ( `<` `adj` $is_adj^ `>` )? ( `(` $parameters^ `)` )?
    (`[` $controls^ (`neg` $negated_qubit_controls^ )? `]`)?
    $targets `:` functional-type(operands, results) attr-dict
  }];

  let hasVerifier = 1;

  code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // MemoryEffects interface
    //===------------------------------------------------------------------===//

    void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
      mlir::MemoryEffects::Effect>> &effects) {
      quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
    }

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    bool isAdj() { return getIsAdj(); }

    //===------------------------------------------------------------------===//
    // Element Access
    //===------------------------------------------------------------------===//

    mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

    mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//

    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    void getOperatorMatrix(Matrix &matrix);
  }];
}

class OneTargetOp<string mnemonic, list<Trait> traits = []> :
    QuakeOperator<mnemonic, !listconcat([NumParameters<0>, NumTargets<1>],
                traits)>;

class OneTargetParamOp<string mnemonic, list<Trait> traits = []> :
    QuakeOperator<mnemonic, !listconcat([NumParameters<1>, NumTargets<1>],
                traits)>;

class TwoTargetOp<string mnemonic, list<Trait> traits = []> :
    QuakeOperator<mnemonic, !listconcat([NumParameters<0>, NumTargets<2>],
                traits)>;

//===----------------------------------------------------------------------===//
// Unitary quantum transformations
//===----------------------------------------------------------------------===//

def quake_ExpPauliOp : QuakeOp<"exp_pauli",
    [QuantumGate, AttrSizedOperandSegments, OperatorInterface,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "General Pauli tensor product rotation";
  let description = [{
    This operation affects a general Pauli tensor product rotation on 
    the input qubits. The number of Pauli characters in the input Pauli word 
    string must equal the number of qubits in the veq. Mathematically, this
    operation applies exp(i theta P) where P is a general Pauli tensor product. 
  }];
  
  let arguments = (ins
    UnitAttr:$is_adj,
    Variadic<AnyFloat>:$parameters,
    Variadic<AnyQType>:$controls,
    Variadic<AnyQTargetType>:$targets,
    OptionalAttr<DenseBoolArrayAttr>:$negated_qubit_controls,
    Optional<AnyTypeOf<[cc_PointerType, cc_CharSpanType]>>:$pauli,
    OptionalAttr<StrAttr>:$pauliLiteral
  );
  let results = (outs
    Variadic<WireType>:$wires
  );

  let assemblyFormat = [{
    ( `<` `adj` $is_adj^ `>` )? ( `(` $parameters^ `)` )?
      ( `[` $controls^ ( `neg` $negated_qubit_controls^ )? `]` )? $targets
      `to` custom<RawString>($pauli, $pauliLiteral) `:`
      functional-type(operands, results) attr-dict
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::UnitAttr":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::DenseBoolArrayAttr":$negates,
                   "mlir::Value":$pauli), [{
      return build($_builder, $_state, mlir::TypeRange{}, is_adj, parameters,
                   controls, targets, negates, pauli, mlir::StringAttr{});
    }]>,
    OpBuilder<(ins "mlir::UnitAttr":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::DenseBoolArrayAttr":$negates,
                   "mlir::StringRef":$pauliLiteral), [{
      return build($_builder, $_state, mlir::TypeRange{}, is_adj, parameters,
                   controls, targets, negates, mlir::Value{},
                   $_builder.getStringAttr(pauliLiteral));
    }]>,
    OpBuilder<(ins "bool":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::Value":$pauli), [{
      return build($_builder, $_state, mlir::TypeRange{}, is_adj, parameters,
                   controls, targets, {}, pauli, mlir::StringAttr{});
    }]>,
    OpBuilder<(ins "bool":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::StringRef":$pauliLiteral), [{
      return build($_builder, $_state, mlir::TypeRange{}, is_adj, parameters,
                   controls, targets, {}, mlir::Value{},
                   $_builder.getStringAttr(pauliLiteral));
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::Value":$pauli), [{
      return build($_builder, $_state, mlir::TypeRange{}, /*is_adj=*/false,
                   parameters, controls, targets, {}, pauli,
                   mlir::StringAttr{});
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::StringRef":$pauliLiteral), [{
      return build($_builder, $_state, mlir::TypeRange{}, /*is_adj=*/false,
                   parameters, controls, targets, {}, mlir::Value{},
                   $_builder.getStringAttr(pauliLiteral));
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::Value":$pauli), [{
      return build($_builder, $_state, mlir::TypeRange{}, /*is_adj=*/false,
                   {}, controls, targets, {}, pauli, mlir::StringAttr{});
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::StringRef":$pauliLiteral), [{
      return build($_builder, $_state, mlir::TypeRange{}, /*is_adj=*/false,
                   {}, controls, targets, {}, mlir::Value{},
                   $_builder.getStringAttr(pauliLiteral));
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$targets,
                   "mlir::Value":$pauli), [{
      return build($_builder, $_state, mlir::TypeRange{}, /*is_adj=*/false,
                   {}, {}, targets, {}, pauli, mlir::StringAttr{});
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$targets,
                   "mlir::StringRef":$pauliLiteral), [{
      return build($_builder, $_state, mlir::TypeRange{}, /*is_adj=*/false,
                   {}, {}, targets, {}, mlir::Value{},
                   $_builder.getStringAttr(pauliLiteral));
    }]>
  ];

  code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // MemoryEffects interface
    //===------------------------------------------------------------------===//

    void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
      mlir::MemoryEffects::Effect>> &effects) {
      quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
    }

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    bool isAdj() { return getIsAdj(); }

    //===------------------------------------------------------------------===//
    // Element Access
    //===------------------------------------------------------------------===//

    mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

    mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//

    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    void getOperatorMatrix(Matrix &matrix) { matrix.clear(); }
  }];
}

def HOp : OneTargetOp<"h", [Hermitian]> {
  let summary = "Hadamard operation";
  let description = [{
    This is a π rotation about the X+Z axis, and has the effect of changing the
    computation basis from |0> (|1>) to |+> (|->) and vice-versa, meaning that
    it enables one to create a superposition of basis states.

    Matrix representation:
    ```
    H = (1 / sqrt(2)) * | 1   1 |
                        | 1  -1 |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ H ├─
     └───┘
    ```
  }];
}

def PhasedRxOp : QuakeOperator<"phased_rx",
                   [NumParameters<2>, NumTargets<1>, Rotation]> {
  let summary = "an arbitrary rotation θ around the cos(φ)x + sin(φ)y axis";
  let description = [{
    Matrix representation:
    ```
    PhasedRx(θ,φ) = |        cos(θ/2)        -iexp(-iφ) * sin(θ/2) |
                    |  -iexp(iφ) * sin(θ/2)         cos(θ/2)       |
    ```

    Circuit symbol:
    ```
     ┌───────────────┐
    ─┤ PhasedRx(θ,φ) ├─
     └───────────────┘
    ```
  }];
}

def R1Op : OneTargetParamOp<"r1", [Rotation]> {
  let summary = "an arbitrary rotation about the |1> state";
  let description = [{
    Matrix representation:
    ```
    R1(λ) = | 1     0    |
            | 0  exp(iλ) |
    ```

    Circuit symbol:
    ```
     ┌───────┐
    ─┤ R1(λ) ├─
     └───────┘
    ```
  }];
}

def RxOp : OneTargetParamOp<"rx", [Rotation]> {
  let summary = "an arbitrary rotation about the X axis";
  let description = [{
    Matrix representation:
    ```
    Rx(θ) = |  cos(θ/2)  -isin(θ/2) |
            | -isin(θ/2)  cos(θ/2)  |
    ```

    Circuit symbol:
    ```
     ┌───────┐
    ─┤ Rx(θ) ├─
     └───────┘
    ```
  }];
  let hasCanonicalizer = 1;
}

def RyOp : OneTargetParamOp<"ry", [Rotation]> {
  let summary = "an arbitrary rotation about the Y axis";
  let description = [{
    Matrix representation:
    ```
    Ry(θ) = | cos(θ/2)  -sin(θ/2) |
            | sin(θ/2)   cos(θ/2) |
    ```

    Circuit symbol:
    ```
     ┌───────┐
    ─┤ Ry(θ) ├─
     └───────┘
    ```
  }];
  let hasCanonicalizer = 1;
}

def RzOp : OneTargetParamOp<"rz", [Rotation]> {
  let summary = "an arbitrary rotation about the Z axis";
  let description = [{
    Matrix representation:
    ```
    Rz(λ) = | exp(-iλ/2)      0     |
            |     0       exp(iλ/2) |
    ```

    Circuit symbol:
    ```
     ┌───────┐
    ─┤ Rz(λ) ├─
     └───────┘
    ```
  }];
  let hasCanonicalizer = 1;
}

def SOp : OneTargetOp<"s"> {
  let summary = "S operation (aka, P or Sqrt(Z))";
  let description = [{
    This operation applies to its target a π/2 rotation about the Z axis.

    Matrix representation:
    ```
    S = | 1   0 |
        | 0   i |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ S ├─
     └───┘
    ```
  }];
}

def SwapOp : TwoTargetOp<"swap", [Hermitian]> {
  let summary = "Swap operation";
  let description = [{
    This operation swaps the states of two qubits.

    Matrix representation:
    ```
    Swap = | 1 0 0 0 |
           | 0 0 1 0 |
           | 0 1 0 0 |
           | 0 0 0 1 |
    ```

    Circuit symbol:
    ```
    ─X─
     │
    ─X─
    ```
  }];
}

def TOp : OneTargetOp<"t"> {
  let summary = "T operation";
  let description = [{
    This operation applies to its target a π/4 rotation about the Z axis.

    Matrix representation:
    ```
    T = | 1      0     |
        | 0  exp(iπ/4) |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ T ├─
     └───┘
    ```
  }];
}

def U2Op : QuakeOperator<"u2", [NumParameters<2>, NumTargets<1>, Rotation]> {
  let summary = "generic rotation about the X+Z axis";
  let description = [{
    The two parameters are Euler angles: φ and λ.

    Matrix representation:
    ```
    U2(φ,λ) = 1/sqrt(2) * | 1        -exp(iλ)       |
                          | exp(iφ)   exp(i(λ + φ)) |
    ```

    Circuit symbol:
    ```
     ┌─────────┐
    ─┤ U2(φ,λ) ├─
     └─────────┘
    ```
  }];
}

def U3Op : QuakeOperator<"u3", [NumParameters<3>, NumTargets<1>, Rotation]> {
  let summary = "the universal three-parameters operator";
  let description = [{
    The three parameters are Euler angles: θ, φ, and λ.

    NOTE: U3 is a generalization of U2 that covers all single-qubit rotations.

    Matrix representation:
    ```
    U3(θ,φ,λ) = | cos(θ/2)            -exp(iλ) * sin(θ/2)       |
                | exp(iφ) * sin(θ/2)   exp(i(λ + φ)) * cos(θ/2) |
    ```

    Circuit symbol:
    ```
     ┌───────────┐
    ─┤ U3(θ,φ,λ) ├─
     └───────────┘
    ```
  }];
}

def XOp : OneTargetOp<"x", [Hermitian]> {
  let summary = "Pauli-X operation (aka, NOT)";
  let description = [{
    Matrix representation:
    ```
    X = | 0  1 |
        | 1  0 |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ X ├─
     └───┘
    ```
  }];
}

def YOp : OneTargetOp<"y", [Hermitian]> {
  let summary = "Pauli-Y operation";
  let description = [{
    Matrix representation:
    ```
    Y = | 0  -i |
        | i   0 |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ Y ├─
     └───┘
    ```
  }];
}

def ZOp : OneTargetOp<"z", [Hermitian]> {
  let summary = "Pauli-Z operation.";
  let description = [{
    Matrix representation:
    ```
    Z = | 1   0 |
        | 0  -1 |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ Z ├─
     └───┘
    ```
  }];
}

def CustomUnitarySymbolOp :
    QuakeOperator<"custom_op", [], (ins SymbolRefAttr:$generator)> {
  let summary = "Custom unitary operation.";
  let description = [{
    Custom unitary operation leveraging a `SymbolRefAttr` describing the
    unitary data generator function. 
  }];

  let builders = [
    OpBuilder<(ins "mlir::SymbolRefAttr":$generator, 
                   "mlir::UnitAttr":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::DenseBoolArrayAttr":$negates), [{
      return build($_builder, $_state, mlir::TypeRange{}, generator, is_adj,
                   parameters, controls, targets, negates);
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$generator,
                   "bool":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::DenseBoolArrayAttr":$negates), [{
      return build($_builder, $_state, mlir::TypeRange{}, generator, is_adj, 
                   parameters, controls, targets, negates);
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$generator,
                   "bool":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, generator, is_adj, parameters, 
                   controls, targets, {});
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$generator,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, generator,  /*is_adj=*/false, 
                   parameters, controls, targets);
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$generator,
                   "bool":$is_adj,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, generator,  is_adj, 
                   mlir::ValueRange{}, controls, targets);
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$generator,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, generator,  /*is_adj=*/false, controls,
                   targets);
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$generator, "bool":$is_adj,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, generator, is_adj, mlir::ValueRange{},
                   targets);
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$generator,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, generator, /*is_adj=*/false, targets);
    }]>
  ];

  let assemblyFormat = [{ $generator
    ( `<` `adj` $is_adj^ `>` )? ( `(` $parameters^ `)` )?
      (`[` $controls^ (`neg` $negated_qubit_controls^ )? `]`)?
      $targets `:` functional-type(operands, results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Quantum states
//===----------------------------------------------------------------------===//

def quake_CreateStateOp : QuakeOp<"create_state", [Pure]> {
  let summary = "Create state from data";
  let description = [{
    This operation takes a pointer to state data and creates a quantum state,
    where state data is a pointer to an array of float or complex numbers.
    The operation can be optimized away in DeleteStates pass, or replaced
    by an intrinsic runtime call on simulators.

    ```mlir
      %0 = quake.create_state %data %len : (!cc.ptr<!cc.array<complex<f64> x 8>>, i64) -> !cc.ptr<!quake.state>
    ```
  }];

  let arguments = (ins
    cc_PointerType:$data,
    AnySignlessInteger:$length
  );
  let results = (outs PointerOf<[quake_StateType]>:$result);
  let assemblyFormat = [{
      $data `,` $length `:` functional-type(operands, results) attr-dict
  }];
}

def QuakeOp_DeleteStateOp : QuakeOp<"delete_state", [] > {
  let summary = "Delete quantum state";
  let description =  [{
    This operation takes a pointer to the state and deletes the state object.
    The operation can be created in in DeleteStates pass, and replaced later
    by an intrinsic runtime call on simulators.

    ```mlir
      quake.delete_state %state : !cc.ptr<!quake.state>
    ```
  }];

  let arguments = (ins PointerOf<[quake_StateType]>:$state);
  let results = (outs);
  let assemblyFormat = [{
      $state `:` type(operands) attr-dict
  }];
}

def quake_GetNumberOfQubitsOp : QuakeOp<"get_number_of_qubits", [Pure] > {
  let summary = "Get number of qubits from a quantum state";
  let description = [{
    This operation takes a pointer to the state as an argument and returns
    a number of qubits in the state. The operation can be optimized away in
    some passes like ReplaceStateByKernel or DeleteStates, or replaced by an
    intrinsic runtime call when the target is one of the simulators.

    ```mlir
      %0 = quake.get_number_of_qubits %state : (!cc.ptr<!quake.state>) -> i64
    ```
  }];

  let arguments = (ins PointerOf<[quake_StateType]>:$state);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = [{
      $state `:` functional-type(operands, results) attr-dict
  }];
}

def QuakeOp_MaterializeStateOp : QuakeOp<"materialize_state", [Pure] > {
  let summary = "Get state from kernel with the provided name.";
  let description = [{
    This operation is created by argument synthesis of state pointer arguments
    for quantum devices.

    It takes two kernel names as symbol references:
      - @num_qubits for determining the size of the allocation to initialize
      - @init for initializing the state the same way as the original kernel
        passed to `cudaq::get_state`.

    This operation will return the state of the original kernel called with
    arguments passed to `cudaq::get_state`.

    The operation may be replaced by calls to the @num_qubits and @init calls,
    which will reproduce the specified state in the `ReplaceStateByKernel`
    pass.

    ```mlir
      %0 = quake.materialize_state @num_qubits, @init : !cc.ptr<!quake.state>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$numQubitsFunc,
    FlatSymbolRefAttr:$initFunc
  );
  let results = (outs PointerOf<[quake_StateType]>:$result);
  let assemblyFormat = [{
     $numQubitsFunc `,` $initFunc `:` qualified(type(results)) attr-dict
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QUAKE_OPS
```

```C++
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace quake {
class CustomUnitarySymbolOp;
} // namespace quake
namespace quake {
class HOp;
} // namespace quake
namespace quake {
class MxOp;
} // namespace quake
namespace quake {
class MyOp;
} // namespace quake
namespace quake {
class MzOp;
} // namespace quake
namespace quake {
class PhasedRxOp;
} // namespace quake
namespace quake {
class DeleteStateOp;
} // namespace quake
namespace quake {
class MaterializeStateOp;
} // namespace quake
namespace quake {
class R1Op;
} // namespace quake
namespace quake {
class RxOp;
} // namespace quake
namespace quake {
class RyOp;
} // namespace quake
namespace quake {
class RzOp;
} // namespace quake
namespace quake {
class SOp;
} // namespace quake
namespace quake {
class SwapOp;
} // namespace quake
namespace quake {
class TOp;
} // namespace quake
namespace quake {
class U2Op;
} // namespace quake
namespace quake {
class U3Op;
} // namespace quake
namespace quake {
class XOp;
} // namespace quake
namespace quake {
class YOp;
} // namespace quake
namespace quake {
class ZOp;
} // namespace quake
namespace quake {
class AllocaOp;
} // namespace quake
namespace quake {
class ApplyNoiseOp;
} // namespace quake
namespace quake {
class ApplyOp;
} // namespace quake
namespace quake {
class BorrowWireOp;
} // namespace quake
namespace quake {
class BundleCableOp;
} // namespace quake
namespace quake {
class ComputeActionOp;
} // namespace quake
namespace quake {
class ConcatOp;
} // namespace quake
namespace quake {
class CreateStateOp;
} // namespace quake
namespace quake {
class DeallocOp;
} // namespace quake
namespace quake {
class DiscriminateOp;
} // namespace quake
namespace quake {
class ExpPauliOp;
} // namespace quake
namespace quake {
class ExtractRefOp;
} // namespace quake
namespace quake {
class FromControlOp;
} // namespace quake
namespace quake {
class GetMemberOp;
} // namespace quake
namespace quake {
class GetNumberOfQubitsOp;
} // namespace quake
namespace quake {
class InitializeStateOp;
} // namespace quake
namespace quake {
class MakeStruqOp;
} // namespace quake
namespace quake {
class NullCableOp;
} // namespace quake
namespace quake {
class NullWireOp;
} // namespace quake
namespace quake {
class RelaxSizeOp;
} // namespace quake
namespace quake {
class ResetOp;
} // namespace quake
namespace quake {
class ReturnWireOp;
} // namespace quake
namespace quake {
class SinkCableOp;
} // namespace quake
namespace quake {
class SinkOp;
} // namespace quake
namespace quake {
class SubVeqOp;
} // namespace quake
namespace quake {
class TerminateCableOp;
} // namespace quake
namespace quake {
class ToControlOp;
} // namespace quake
namespace quake {
class UnwrapOp;
} // namespace quake
namespace quake {
class VeqSizeOp;
} // namespace quake
namespace quake {
class WireSetOp;
} // namespace quake
namespace quake {
class WrapOp;
} // namespace quake
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::CustomUnitarySymbolOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CustomUnitarySymbolOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CustomUnitarySymbolOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::SymbolRefAttr getGeneratorAttr();
  ::mlir::SymbolRefAttr getGenerator();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class CustomUnitarySymbolOpGenericAdaptor : public detail::CustomUnitarySymbolOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CustomUnitarySymbolOpGenericAdaptorBase;
public:
  CustomUnitarySymbolOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CustomUnitarySymbolOpAdaptor : public CustomUnitarySymbolOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CustomUnitarySymbolOpGenericAdaptor::CustomUnitarySymbolOpGenericAdaptor;
  CustomUnitarySymbolOpAdaptor(CustomUnitarySymbolOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CustomUnitarySymbolOp : public ::mlir::Op<CustomUnitarySymbolOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CustomUnitarySymbolOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CustomUnitarySymbolOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("generator"), ::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getGeneratorAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getGeneratorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.custom_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::SymbolRefAttr getGeneratorAttr();
  ::mlir::SymbolRefAttr getGenerator();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setGeneratorAttr(::mlir::SymbolRefAttr attr);
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::SymbolRefAttr generator, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::SymbolRefAttr generator, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::SymbolRefAttr generator, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::SymbolRefAttr generator, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::SymbolRefAttr generator, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::SymbolRefAttr generator, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::SymbolRefAttr generator, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::SymbolRefAttr generator, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, ::mlir::SymbolRefAttr generator, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, ::mlir::SymbolRefAttr generator, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::CustomUnitarySymbolOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::HOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class HOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  HOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class HOpGenericAdaptor : public detail::HOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::HOpGenericAdaptorBase;
public:
  HOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class HOpAdaptor : public HOpGenericAdaptor<::mlir::ValueRange> {
public:
  using HOpGenericAdaptor::HOpGenericAdaptor;
  HOpAdaptor(HOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class HOp : public ::mlir::Op<HOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Hermitian> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = HOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = HOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.h");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::HOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::MxOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MxOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  MxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getRegisterNameAttr();
  ::std::optional< ::llvm::StringRef > getRegisterName();
};
} // namespace detail
template <typename RangeT>
class MxOpGenericAdaptor : public detail::MxOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MxOpGenericAdaptorBase;
public:
  MxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getTargets() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MxOpAdaptor : public MxOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MxOpGenericAdaptor::MxOpGenericAdaptor;
  MxOpAdaptor(MxOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class MxOp : public ::mlir::Op<MxOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, quake::MeasurementInterface::Trait, ::cudaq::QuantumMeasure, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MxOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MxOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("registerName")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getRegisterNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getRegisterNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.mx");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getMeasOut();
  ::mlir::Operation::result_range getWires();
  ::mlir::StringAttr getRegisterNameAttr();
  ::std::optional< ::llvm::StringRef > getRegisterName();
  void setRegisterNameAttr(::mlir::StringAttr attr);
  void setRegisterName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeRegisterNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type measOut, ::mlir::TypeRange wires, ::mlir::ValueRange targets, /*optional*/::mlir::StringAttr registerName);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange targets, /*optional*/::mlir::StringAttr registerName);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getMeasurementEffectsImpl(effects, getTargets());
  }
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::MxOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::MyOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MyOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  MyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getRegisterNameAttr();
  ::std::optional< ::llvm::StringRef > getRegisterName();
};
} // namespace detail
template <typename RangeT>
class MyOpGenericAdaptor : public detail::MyOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MyOpGenericAdaptorBase;
public:
  MyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getTargets() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MyOpAdaptor : public MyOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MyOpGenericAdaptor::MyOpGenericAdaptor;
  MyOpAdaptor(MyOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class MyOp : public ::mlir::Op<MyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, quake::MeasurementInterface::Trait, ::cudaq::QuantumMeasure, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MyOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MyOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("registerName")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getRegisterNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getRegisterNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.my");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getMeasOut();
  ::mlir::Operation::result_range getWires();
  ::mlir::StringAttr getRegisterNameAttr();
  ::std::optional< ::llvm::StringRef > getRegisterName();
  void setRegisterNameAttr(::mlir::StringAttr attr);
  void setRegisterName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeRegisterNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type measOut, ::mlir::TypeRange wires, ::mlir::ValueRange targets, /*optional*/::mlir::StringAttr registerName);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange targets, /*optional*/::mlir::StringAttr registerName);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getMeasurementEffectsImpl(effects, getTargets());
  }
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::MyOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::MzOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MzOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  MzOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getRegisterNameAttr();
  ::std::optional< ::llvm::StringRef > getRegisterName();
};
} // namespace detail
template <typename RangeT>
class MzOpGenericAdaptor : public detail::MzOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MzOpGenericAdaptorBase;
public:
  MzOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getTargets() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MzOpAdaptor : public MzOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MzOpGenericAdaptor::MzOpGenericAdaptor;
  MzOpAdaptor(MzOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class MzOp : public ::mlir::Op<MzOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, quake::MeasurementInterface::Trait, ::cudaq::QuantumMeasure, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MzOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MzOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("registerName")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getRegisterNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getRegisterNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.mz");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getMeasOut();
  ::mlir::Operation::result_range getWires();
  ::mlir::StringAttr getRegisterNameAttr();
  ::std::optional< ::llvm::StringRef > getRegisterName();
  void setRegisterNameAttr(::mlir::StringAttr attr);
  void setRegisterName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeRegisterNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type measOut, ::mlir::TypeRange wires, ::mlir::ValueRange targets, /*optional*/::mlir::StringAttr registerName);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange targets, /*optional*/::mlir::StringAttr registerName);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getMeasurementEffectsImpl(effects, getTargets());
  }
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::MzOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::PhasedRxOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class PhasedRxOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  PhasedRxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class PhasedRxOpGenericAdaptor : public detail::PhasedRxOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PhasedRxOpGenericAdaptorBase;
public:
  PhasedRxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class PhasedRxOpAdaptor : public PhasedRxOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PhasedRxOpGenericAdaptor::PhasedRxOpGenericAdaptor;
  PhasedRxOpAdaptor(PhasedRxOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class PhasedRxOp : public ::mlir::Op<PhasedRxOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Rotation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PhasedRxOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PhasedRxOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.phased_rx");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::PhasedRxOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::DeleteStateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DeleteStateOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DeleteStateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class DeleteStateOpGenericAdaptor : public detail::DeleteStateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DeleteStateOpGenericAdaptorBase;
public:
  DeleteStateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getState() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DeleteStateOpAdaptor : public DeleteStateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DeleteStateOpGenericAdaptor::DeleteStateOpGenericAdaptor;
  DeleteStateOpAdaptor(DeleteStateOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DeleteStateOp : public ::mlir::Op<DeleteStateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DeleteStateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DeleteStateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.delete_state");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::cudaq::cc::PointerType> getState();
  ::mlir::MutableOperandRange getStateMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value state);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value state);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::DeleteStateOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::MaterializeStateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MaterializeStateOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  MaterializeStateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr getNumQubitsFuncAttr();
  ::llvm::StringRef getNumQubitsFunc();
  ::mlir::FlatSymbolRefAttr getInitFuncAttr();
  ::llvm::StringRef getInitFunc();
};
} // namespace detail
template <typename RangeT>
class MaterializeStateOpGenericAdaptor : public detail::MaterializeStateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MaterializeStateOpGenericAdaptorBase;
public:
  MaterializeStateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MaterializeStateOpAdaptor : public MaterializeStateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MaterializeStateOpGenericAdaptor::MaterializeStateOpGenericAdaptor;
  MaterializeStateOpAdaptor(MaterializeStateOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class MaterializeStateOp : public ::mlir::Op<MaterializeStateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::cudaq::cc::PointerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MaterializeStateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MaterializeStateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("initFunc"), ::llvm::StringRef("numQubitsFunc")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getInitFuncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInitFuncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNumQubitsFuncAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNumQubitsFuncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.materialize_state");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::cudaq::cc::PointerType> getResult();
  ::mlir::FlatSymbolRefAttr getNumQubitsFuncAttr();
  ::llvm::StringRef getNumQubitsFunc();
  ::mlir::FlatSymbolRefAttr getInitFuncAttr();
  ::llvm::StringRef getInitFunc();
  void setNumQubitsFuncAttr(::mlir::FlatSymbolRefAttr attr);
  void setNumQubitsFunc(::llvm::StringRef attrValue);
  void setInitFuncAttr(::mlir::FlatSymbolRefAttr attr);
  void setInitFunc(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::FlatSymbolRefAttr numQubitsFunc, ::mlir::FlatSymbolRefAttr initFunc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr numQubitsFunc, ::mlir::FlatSymbolRefAttr initFunc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef numQubitsFunc, ::llvm::StringRef initFunc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef numQubitsFunc, ::llvm::StringRef initFunc);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::MaterializeStateOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::R1Op declarations
//===----------------------------------------------------------------------===//

namespace detail {
class R1OpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  R1OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class R1OpGenericAdaptor : public detail::R1OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::R1OpGenericAdaptorBase;
public:
  R1OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class R1OpAdaptor : public R1OpGenericAdaptor<::mlir::ValueRange> {
public:
  using R1OpGenericAdaptor::R1OpGenericAdaptor;
  R1OpAdaptor(R1Op op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class R1Op : public ::mlir::Op<R1Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Rotation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = R1OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = R1OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.r1");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::R1Op)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::RxOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RxOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class RxOpGenericAdaptor : public detail::RxOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RxOpGenericAdaptorBase;
public:
  RxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RxOpAdaptor : public RxOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RxOpGenericAdaptor::RxOpGenericAdaptor;
  RxOpAdaptor(RxOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RxOp : public ::mlir::Op<RxOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Rotation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RxOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RxOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.rx");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::RxOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::RyOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RyOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class RyOpGenericAdaptor : public detail::RyOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RyOpGenericAdaptorBase;
public:
  RyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RyOpAdaptor : public RyOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RyOpGenericAdaptor::RyOpGenericAdaptor;
  RyOpAdaptor(RyOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RyOp : public ::mlir::Op<RyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Rotation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RyOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RyOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.ry");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::RyOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::RzOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RzOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RzOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class RzOpGenericAdaptor : public detail::RzOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RzOpGenericAdaptorBase;
public:
  RzOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RzOpAdaptor : public RzOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RzOpGenericAdaptor::RzOpGenericAdaptor;
  RzOpAdaptor(RzOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RzOp : public ::mlir::Op<RzOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Rotation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RzOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RzOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.rz");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::RzOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::SOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class SOpGenericAdaptor : public detail::SOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SOpGenericAdaptorBase;
public:
  SOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SOpAdaptor : public SOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SOpGenericAdaptor::SOpGenericAdaptor;
  SOpAdaptor(SOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SOp : public ::mlir::Op<SOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.s");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::SOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::SwapOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SwapOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SwapOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class SwapOpGenericAdaptor : public detail::SwapOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SwapOpGenericAdaptorBase;
public:
  SwapOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SwapOpAdaptor : public SwapOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SwapOpGenericAdaptor::SwapOpGenericAdaptor;
  SwapOpAdaptor(SwapOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SwapOp : public ::mlir::Op<SwapOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Hermitian> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SwapOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SwapOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.swap");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::SwapOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::TOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class TOpGenericAdaptor : public detail::TOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TOpGenericAdaptorBase;
public:
  TOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TOpAdaptor : public TOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TOpGenericAdaptor::TOpGenericAdaptor;
  TOpAdaptor(TOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TOp : public ::mlir::Op<TOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.t");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::TOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::U2Op declarations
//===----------------------------------------------------------------------===//

namespace detail {
class U2OpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  U2OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class U2OpGenericAdaptor : public detail::U2OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::U2OpGenericAdaptorBase;
public:
  U2OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class U2OpAdaptor : public U2OpGenericAdaptor<::mlir::ValueRange> {
public:
  using U2OpGenericAdaptor::U2OpGenericAdaptor;
  U2OpAdaptor(U2Op op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class U2Op : public ::mlir::Op<U2Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Rotation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = U2OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = U2OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.u2");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::U2Op)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::U3Op declarations
//===----------------------------------------------------------------------===//

namespace detail {
class U3OpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  U3OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class U3OpGenericAdaptor : public detail::U3OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::U3OpGenericAdaptorBase;
public:
  U3OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class U3OpAdaptor : public U3OpGenericAdaptor<::mlir::ValueRange> {
public:
  using U3OpGenericAdaptor::U3OpGenericAdaptor;
  U3OpAdaptor(U3Op op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class U3Op : public ::mlir::Op<U3Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Rotation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = U3OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = U3OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.u3");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::U3Op)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::XOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class XOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  XOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class XOpGenericAdaptor : public detail::XOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::XOpGenericAdaptorBase;
public:
  XOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class XOpAdaptor : public XOpGenericAdaptor<::mlir::ValueRange> {
public:
  using XOpGenericAdaptor::XOpGenericAdaptor;
  XOpAdaptor(XOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class XOp : public ::mlir::Op<XOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Hermitian> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = XOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = XOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.x");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::XOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::YOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class YOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  YOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class YOpGenericAdaptor : public detail::YOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::YOpGenericAdaptorBase;
public:
  YOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class YOpAdaptor : public YOpGenericAdaptor<::mlir::ValueRange> {
public:
  using YOpGenericAdaptor::YOpGenericAdaptor;
  YOpAdaptor(YOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class YOp : public ::mlir::Op<YOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Hermitian> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = YOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.y");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::YOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::ZOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ZOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ZOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
};
} // namespace detail
template <typename RangeT>
class ZOpGenericAdaptor : public detail::ZOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ZOpGenericAdaptorBase;
public:
  ZOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ZOpAdaptor : public ZOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ZOpGenericAdaptor::ZOpGenericAdaptor;
  ZOpAdaptor(ZOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ZOp : public ::mlir::Op<ZOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::cudaq::Hermitian> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ZOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ZOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.z");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix);
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::ZOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::AllocaOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllocaOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AllocaOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AllocaOpGenericAdaptor : public detail::AllocaOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllocaOpGenericAdaptorBase;
public:
  AllocaOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSize() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : *operands.begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllocaOpAdaptor : public AllocaOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllocaOpGenericAdaptor::AllocaOpGenericAdaptor;
  AllocaOpAdaptor(AllocaOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllocaOp : public ::mlir::Op<AllocaOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocaOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllocaOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.alloca");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getSize();
  ::mlir::MutableOperandRange getSizeMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getRefOrVec();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, size_t size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Type ty);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type ref_or_vec, /*optional*/::mlir::Value size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value size);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  bool hasInitializedState() {
    auto *self = getOperation();
    return self->hasOneUse() &&
      mlir::isa<quake::InitializeStateOp>(*self->getUsers().begin());
  }

  quake::InitializeStateOp getInitializedState();
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::AllocaOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::ApplyNoiseOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ApplyNoiseOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ApplyNoiseOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr getNoiseFuncAttr();
  ::std::optional< ::llvm::StringRef > getNoiseFunc();
};
} // namespace detail
template <typename RangeT>
class ApplyNoiseOpGenericAdaptor : public detail::ApplyNoiseOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ApplyNoiseOpGenericAdaptorBase;
public:
  ApplyNoiseOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getKey() {
    auto operands = getODSOperands(0);
    return operands.empty() ? ValueT{} : *operands.begin();
  }

  RangeT getParameters() {
    return getODSOperands(1);
  }

  RangeT getQubits() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ApplyNoiseOpAdaptor : public ApplyNoiseOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ApplyNoiseOpGenericAdaptor::ApplyNoiseOpGenericAdaptor;
  ApplyNoiseOpAdaptor(ApplyNoiseOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ApplyNoiseOp : public ::mlir::Op<ApplyNoiseOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ApplyNoiseOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ApplyNoiseOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("noise_func"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNoiseFuncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNoiseFuncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.apply_noise");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getKey();
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getQubits();
  ::mlir::MutableOperandRange getKeyMutable();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getQubitsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::FlatSymbolRefAttr getNoiseFuncAttr();
  ::std::optional< ::llvm::StringRef > getNoiseFunc();
  void setNoiseFuncAttr(::mlir::FlatSymbolRefAttr attr);
  void setNoiseFunc(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeNoiseFuncAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::StringRef noise_func, mlir::ValueRange parameters, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::FlatSymbolRefAttr noise_func, mlir::ValueRange parameters, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value key, mlir::ValueRange parameters, mlir::ValueRange targets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::FlatSymbolRefAttr noise_func, /*optional*/::mlir::Value key, ::mlir::ValueRange parameters, ::mlir::ValueRange qubits);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::FlatSymbolRefAttr noise_func, /*optional*/::mlir::Value key, ::mlir::ValueRange parameters, ::mlir::ValueRange qubits);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static constexpr mlir::StringRef getNoiseFuncAttrNameStr() {
    return "noise_func";
  }
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::ApplyNoiseOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::ApplyOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ApplyOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ApplyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::SymbolRefAttr getCalleeAttr();
  ::std::optional< ::mlir::SymbolRefAttr > getCallee();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
};
} // namespace detail
template <typename RangeT>
class ApplyOpGenericAdaptor : public detail::ApplyOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ApplyOpGenericAdaptorBase;
public:
  ApplyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getIndirectCallee() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getArgs() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ApplyOpAdaptor : public ApplyOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ApplyOpGenericAdaptor::ApplyOpGenericAdaptor;
  ApplyOpAdaptor(ApplyOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ApplyOp : public ::mlir::Op<ApplyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::CallOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ApplyOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ApplyOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("callee"), ::llvm::StringRef("is_adj"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCalleeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCalleeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.apply");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getIndirectCallee();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getIndirectCalleeMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::SymbolRefAttr getCalleeAttr();
  ::std::optional< ::mlir::SymbolRefAttr > getCallee();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  void setCalleeAttr(::mlir::SymbolRefAttr attr);
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  ::mlir::Attribute removeCalleeAttr();
  ::mlir::Attribute removeIsAdjAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::TypeRange retTy, mlir::SymbolRefAttr callee, mlir::UnitAttr is_adj, mlir::ValueRange controls, mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::TypeRange retTy, mlir::SymbolRefAttr callee, bool is_adj, mlir::ValueRange controls, mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::TypeRange retTy, mlir::Value callable, mlir::UnitAttr is_adj, mlir::ValueRange controls, mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::TypeRange retTy, mlir::Value callable, bool is_adj, mlir::ValueRange controls, mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, /*optional*/::mlir::SymbolRefAttr callee, ::mlir::ValueRange indirect_callee, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange controls, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, /*optional*/::mlir::SymbolRefAttr callee, ::mlir::ValueRange indirect_callee, /*optional*/bool is_adj, ::mlir::ValueRange controls, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static constexpr llvm::StringRef getCalleeAttrNameStr() { return "callee"; }

  mlir::FunctionType getFunctionType();

  /// Get the argument operands to the called function.
  operand_range getArgOperands() {
    if (getControls().empty())
      return {operand_begin(), operand_end()};
    return {getArgs().begin(), getArgs().end()};
  }

  bool applyToVariant() {
    return getIsAdj() || !getControls().empty();
  }

  /// Return the callee of this operation.
  mlir::CallInterfaceCallable getCallableForCallee() {
    return (*this)->getAttrOfType<mlir::SymbolRefAttr>(getCalleeAttrName());
  }
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::ApplyOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::BorrowWireOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class BorrowWireOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  BorrowWireOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr getSetNameAttr();
  ::llvm::StringRef getSetName();
  ::mlir::IntegerAttr getIdentityAttr();
  uint32_t getIdentity();
};
} // namespace detail
template <typename RangeT>
class BorrowWireOpGenericAdaptor : public detail::BorrowWireOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BorrowWireOpGenericAdaptorBase;
public:
  BorrowWireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BorrowWireOpAdaptor : public BorrowWireOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BorrowWireOpGenericAdaptor::BorrowWireOpGenericAdaptor;
  BorrowWireOpAdaptor(BorrowWireOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class BorrowWireOp : public ::mlir::Op<BorrowWireOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::WireType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BorrowWireOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BorrowWireOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("identity"), ::llvm::StringRef("set_name")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIdentityAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIdentityAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSetNameAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSetNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.borrow_wire");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::FlatSymbolRefAttr getSetNameAttr();
  ::llvm::StringRef getSetName();
  ::mlir::IntegerAttr getIdentityAttr();
  uint32_t getIdentity();
  void setSetNameAttr(::mlir::FlatSymbolRefAttr attr);
  void setSetName(::llvm::StringRef attrValue);
  void setIdentityAttr(::mlir::IntegerAttr attr);
  void setIdentity(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::FlatSymbolRefAttr set_name, ::mlir::IntegerAttr identity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr set_name, ::mlir::IntegerAttr identity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef set_name, uint32_t identity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef set_name, uint32_t identity);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::BorrowWireOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::BundleCableOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class BundleCableOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  BundleCableOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class BundleCableOpGenericAdaptor : public detail::BundleCableOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BundleCableOpGenericAdaptorBase;
public:
  BundleCableOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getWires() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BundleCableOpAdaptor : public BundleCableOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BundleCableOpGenericAdaptor::BundleCableOpGenericAdaptor;
  BundleCableOpAdaptor(BundleCableOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class BundleCableOp : public ::mlir::Op<BundleCableOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::CableType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BundleCableOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BundleCableOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.bundle_cable");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getWires();
  ::mlir::MutableOperandRange getWiresMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::ValueRange wires);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::BundleCableOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::ComputeActionOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ComputeActionOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ComputeActionOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsDaggerAttr();
  bool getIsDagger();
};
} // namespace detail
template <typename RangeT>
class ComputeActionOpGenericAdaptor : public detail::ComputeActionOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ComputeActionOpGenericAdaptorBase;
public:
  ComputeActionOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getCompute() {
    return *getODSOperands(0).begin();
  }

  ValueT getAction() {
    return *getODSOperands(1).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ComputeActionOpAdaptor : public ComputeActionOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ComputeActionOpGenericAdaptor::ComputeActionOpGenericAdaptor;
  ComputeActionOpAdaptor(ComputeActionOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ComputeActionOp : public ::mlir::Op<ComputeActionOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ComputeActionOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ComputeActionOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_dagger")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsDaggerAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsDaggerAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.compute_action");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getCompute();
  ::mlir::Value getAction();
  ::mlir::MutableOperandRange getComputeMutable();
  ::mlir::MutableOperandRange getActionMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr getIsDaggerAttr();
  bool getIsDagger();
  void setIsDaggerAttr(::mlir::UnitAttr attr);
  void setIsDagger(bool attrValue);
  ::mlir::Attribute removeIsDaggerAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr is_dagger, ::mlir::Value compute, ::mlir::Value action);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr is_dagger, ::mlir::Value compute, ::mlir::Value action);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool is_dagger, ::mlir::Value compute, ::mlir::Value action);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool is_dagger, ::mlir::Value compute, ::mlir::Value action);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::ComputeActionOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::ConcatOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ConcatOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ConcatOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ConcatOpGenericAdaptor : public detail::ConcatOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConcatOpGenericAdaptorBase;
public:
  ConcatOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getQbits() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConcatOpAdaptor : public ConcatOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConcatOpGenericAdaptor::ConcatOpGenericAdaptor;
  ConcatOpAdaptor(ConcatOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ConcatOp : public ::mlir::Op<ConcatOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::VeqType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConcatOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConcatOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.concat");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getQbits();
  ::mlir::MutableOperandRange getQbitsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::ValueRange qbits);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::ConcatOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::CreateStateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CreateStateOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CreateStateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class CreateStateOpGenericAdaptor : public detail::CreateStateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CreateStateOpGenericAdaptorBase;
public:
  CreateStateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getData() {
    return *getODSOperands(0).begin();
  }

  ValueT getLength() {
    return *getODSOperands(1).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CreateStateOpAdaptor : public CreateStateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CreateStateOpGenericAdaptor::CreateStateOpGenericAdaptor;
  CreateStateOpAdaptor(CreateStateOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CreateStateOp : public ::mlir::Op<CreateStateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::cudaq::cc::PointerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CreateStateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CreateStateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.create_state");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getData();
  ::mlir::TypedValue<::mlir::IntegerType> getLength();
  ::mlir::MutableOperandRange getDataMutable();
  ::mlir::MutableOperandRange getLengthMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::cudaq::cc::PointerType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value data, ::mlir::Value length);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value length);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::CreateStateOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::DeallocOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DeallocOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DeallocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class DeallocOpGenericAdaptor : public detail::DeallocOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DeallocOpGenericAdaptorBase;
public:
  DeallocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getReference() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DeallocOpAdaptor : public DeallocOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DeallocOpGenericAdaptor::DeallocOpGenericAdaptor;
  DeallocOpAdaptor(DeallocOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DeallocOp : public ::mlir::Op<DeallocOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DeallocOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DeallocOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.dealloc");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getReference();
  ::mlir::MutableOperandRange getReferenceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value reference);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value reference);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::DeallocOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::DiscriminateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DiscriminateOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DiscriminateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class DiscriminateOpGenericAdaptor : public detail::DiscriminateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DiscriminateOpGenericAdaptorBase;
public:
  DiscriminateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMeasurement() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DiscriminateOpAdaptor : public DiscriminateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DiscriminateOpGenericAdaptor::DiscriminateOpGenericAdaptor;
  DiscriminateOpAdaptor(DiscriminateOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DiscriminateOp : public ::mlir::Op<DiscriminateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DiscriminateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DiscriminateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.discriminate");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getMeasurement();
  ::mlir::MutableOperandRange getMeasurementMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value measurement);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value measurement);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::DiscriminateOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::ExpPauliOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ExpPauliOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ExpPauliOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  ::mlir::StringAttr getPauliLiteralAttr();
  ::std::optional< ::llvm::StringRef > getPauliLiteral();
};
} // namespace detail
template <typename RangeT>
class ExpPauliOpGenericAdaptor : public detail::ExpPauliOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExpPauliOpGenericAdaptorBase;
public:
  ExpPauliOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getParameters() {
    return getODSOperands(0);
  }

  RangeT getControls() {
    return getODSOperands(1);
  }

  RangeT getTargets() {
    return getODSOperands(2);
  }

  ValueT getPauli() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : *operands.begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExpPauliOpAdaptor : public ExpPauliOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExpPauliOpGenericAdaptor::ExpPauliOpGenericAdaptor;
  ExpPauliOpAdaptor(ExpPauliOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ExpPauliOp : public ::mlir::Op<ExpPauliOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::quake::OperatorInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExpPauliOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExpPauliOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_adj"), ::llvm::StringRef("negated_qubit_controls"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("pauliLiteral")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsAdjAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsAdjAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNegatedQubitControlsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNegatedQubitControlsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getPauliLiteralAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getPauliLiteralAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.exp_pauli");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getParameters();
  ::mlir::Operation::operand_range getControls();
  ::mlir::Operation::operand_range getTargets();
  ::mlir::Value getPauli();
  ::mlir::MutableOperandRange getParametersMutable();
  ::mlir::MutableOperandRange getControlsMutable();
  ::mlir::MutableOperandRange getTargetsMutable();
  ::mlir::MutableOperandRange getPauliMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  ::mlir::UnitAttr getIsAdjAttr();
  bool getIsAdj();
  ::mlir::DenseBoolArrayAttr getNegatedQubitControlsAttr();
  ::std::optional<::llvm::ArrayRef<bool>> getNegatedQubitControls();
  ::mlir::StringAttr getPauliLiteralAttr();
  ::std::optional< ::llvm::StringRef > getPauliLiteral();
  void setIsAdjAttr(::mlir::UnitAttr attr);
  void setIsAdj(bool attrValue);
  void setNegatedQubitControlsAttr(::mlir::DenseBoolArrayAttr attr);
  void setNegatedQubitControls(::std::optional<::llvm::ArrayRef<bool>> attrValue);
  void setPauliLiteralAttr(::mlir::StringAttr attr);
  void setPauliLiteral(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeIsAdjAttr();
  ::mlir::Attribute removeNegatedQubitControlsAttr();
  ::mlir::Attribute removePauliLiteralAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates, mlir::Value pauli);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::UnitAttr is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::DenseBoolArrayAttr negates, mlir::StringRef pauliLiteral);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::Value pauli);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool is_adj, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::StringRef pauliLiteral);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::Value pauli);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange parameters, mlir::ValueRange controls, mlir::ValueRange targets, mlir::StringRef pauliLiteral);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets, mlir::Value pauli);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange controls, mlir::ValueRange targets, mlir::StringRef pauliLiteral);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets, mlir::Value pauli);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::ValueRange targets, mlir::StringRef pauliLiteral);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/::mlir::UnitAttr is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls, /*optional*/::mlir::Value pauli, /*optional*/::mlir::StringAttr pauliLiteral);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, /*optional*/bool is_adj, ::mlir::ValueRange parameters, ::mlir::ValueRange controls, ::mlir::ValueRange targets, /*optional*/::mlir::DenseBoolArrayAttr negated_qubit_controls, /*optional*/::mlir::Value pauli, /*optional*/::mlir::StringAttr pauliLiteral);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // MemoryEffects interface
  //===------------------------------------------------------------------===//

  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
    mlir::MemoryEffects::Effect>> &effects) {
    quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
  }

  //===------------------------------------------------------------------===//
  // Properties
  //===------------------------------------------------------------------===//

  bool isAdj() { return getIsAdj(); }

  //===------------------------------------------------------------------===//
  // Element Access
  //===------------------------------------------------------------------===//

  mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

  mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

  //===------------------------------------------------------------------===//
  // Operator interface
  //===------------------------------------------------------------------===//

  using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

  void getOperatorMatrix(Matrix &matrix) { matrix.clear(); }
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::ExpPauliOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::ExtractRefOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ExtractRefOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ExtractRefOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getRawIndexAttr();
  uint64_t getRawIndex();
};
} // namespace detail
template <typename RangeT>
class ExtractRefOpGenericAdaptor : public detail::ExtractRefOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExtractRefOpGenericAdaptorBase;
public:
  ExtractRefOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVeq() {
    return *getODSOperands(0).begin();
  }

  ValueT getIndex() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : *operands.begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExtractRefOpAdaptor : public ExtractRefOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExtractRefOpGenericAdaptor::ExtractRefOpGenericAdaptor;
  ExtractRefOpAdaptor(ExtractRefOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ExtractRefOp : public ::mlir::Op<ExtractRefOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::RefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExtractRefOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExtractRefOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("rawIndex")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getRawIndexAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getRawIndexAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.extract_ref");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::VeqType> getVeq();
  ::mlir::Value getIndex();
  ::mlir::MutableOperandRange getVeqMutable();
  ::mlir::MutableOperandRange getIndexMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::quake::RefType> getRef();
  ::mlir::IntegerAttr getRawIndexAttr();
  uint64_t getRawIndex();
  void setRawIndexAttr(::mlir::IntegerAttr attr);
  void setRawIndex(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value veq, mlir::Value index, mlir::IntegerAttr rawIndex);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value veq, mlir::Value index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value veq, std::size_t rawIndex);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type ref, ::mlir::Value veq, /*optional*/::mlir::Value index, ::mlir::IntegerAttr rawIndex);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value veq, /*optional*/::mlir::Value index, ::mlir::IntegerAttr rawIndex);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type ref, ::mlir::Value veq, /*optional*/::mlir::Value index, uint64_t rawIndex);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value veq, /*optional*/::mlir::Value index, uint64_t rawIndex);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static constexpr std::size_t kDynamicIndex =
    std::numeric_limits<std::size_t>::max();

  bool hasConstantIndex() { return !getIndex(); }
  std::size_t getConstantIndex() { return getRawIndex(); }
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::ExtractRefOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::FromControlOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class FromControlOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  FromControlOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class FromControlOpGenericAdaptor : public detail::FromControlOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::FromControlOpGenericAdaptorBase;
public:
  FromControlOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getCtrlbit() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class FromControlOpAdaptor : public FromControlOpGenericAdaptor<::mlir::ValueRange> {
public:
  using FromControlOpGenericAdaptor::FromControlOpGenericAdaptor;
  FromControlOpAdaptor(FromControlOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class FromControlOp : public ::mlir::Op<FromControlOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::WireType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FromControlOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = FromControlOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.from_ctrl");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::ControlType> getCtrlbit();
  ::mlir::MutableOperandRange getCtrlbitMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value ctrlbit);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ctrlbit);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::FromControlOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::GetMemberOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GetMemberOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GetMemberOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getIndexAttr();
  uint32_t getIndex();
};
} // namespace detail
template <typename RangeT>
class GetMemberOpGenericAdaptor : public detail::GetMemberOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GetMemberOpGenericAdaptorBase;
public:
  GetMemberOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getStruq() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GetMemberOpAdaptor : public GetMemberOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GetMemberOpGenericAdaptor::GetMemberOpGenericAdaptor;
  GetMemberOpAdaptor(GetMemberOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GetMemberOp : public ::mlir::Op<GetMemberOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GetMemberOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GetMemberOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("index")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.get_member");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::StruqType> getStruq();
  ::mlir::MutableOperandRange getStruqMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr getIndexAttr();
  uint32_t getIndex();
  void setIndexAttr(::mlir::IntegerAttr attr);
  void setIndex(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value struq, ::mlir::IntegerAttr index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value struq, ::mlir::IntegerAttr index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value struq, uint32_t index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value struq, uint32_t index);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::GetMemberOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::GetNumberOfQubitsOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GetNumberOfQubitsOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GetNumberOfQubitsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class GetNumberOfQubitsOpGenericAdaptor : public detail::GetNumberOfQubitsOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GetNumberOfQubitsOpGenericAdaptorBase;
public:
  GetNumberOfQubitsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getState() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GetNumberOfQubitsOpAdaptor : public GetNumberOfQubitsOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GetNumberOfQubitsOpGenericAdaptor::GetNumberOfQubitsOpGenericAdaptor;
  GetNumberOfQubitsOpAdaptor(GetNumberOfQubitsOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GetNumberOfQubitsOp : public ::mlir::Op<GetNumberOfQubitsOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GetNumberOfQubitsOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GetNumberOfQubitsOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.get_number_of_qubits");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::cudaq::cc::PointerType> getState();
  ::mlir::MutableOperandRange getStateMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value state);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value state);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::GetNumberOfQubitsOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::InitializeStateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class InitializeStateOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  InitializeStateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class InitializeStateOpGenericAdaptor : public detail::InitializeStateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::InitializeStateOpGenericAdaptorBase;
public:
  InitializeStateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getTargets() {
    return *getODSOperands(0).begin();
  }

  ValueT getState() {
    return *getODSOperands(1).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class InitializeStateOpAdaptor : public InitializeStateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using InitializeStateOpGenericAdaptor::InitializeStateOpGenericAdaptor;
  InitializeStateOpAdaptor(InitializeStateOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class InitializeStateOp : public ::mlir::Op<InitializeStateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::VeqType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InitializeStateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = InitializeStateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.init_state");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::VeqType> getTargets();
  ::mlir::Value getState();
  ::mlir::MutableOperandRange getTargetsMutable();
  ::mlir::MutableOperandRange getStateMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value targets, ::mlir::Value state);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value targets, ::mlir::Value state);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::InitializeStateOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::MakeStruqOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MakeStruqOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  MakeStruqOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class MakeStruqOpGenericAdaptor : public detail::MakeStruqOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MakeStruqOpGenericAdaptorBase;
public:
  MakeStruqOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getVeqs() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MakeStruqOpAdaptor : public MakeStruqOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MakeStruqOpGenericAdaptor::MakeStruqOpGenericAdaptor;
  MakeStruqOpAdaptor(MakeStruqOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class MakeStruqOp : public ::mlir::Op<MakeStruqOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::StruqType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MakeStruqOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MakeStruqOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.make_struq");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getVeqs();
  ::mlir::MutableOperandRange getVeqsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::ValueRange veqs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::MakeStruqOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::NullCableOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class NullCableOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  NullCableOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class NullCableOpGenericAdaptor : public detail::NullCableOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::NullCableOpGenericAdaptorBase;
public:
  NullCableOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class NullCableOpAdaptor : public NullCableOpGenericAdaptor<::mlir::ValueRange> {
public:
  using NullCableOpGenericAdaptor::NullCableOpGenericAdaptor;
  NullCableOpAdaptor(NullCableOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class NullCableOp : public ::mlir::Op<NullCableOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::CableType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = NullCableOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = NullCableOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.null_cable");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::quake::CableType> getCableType();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type cableType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::NullCableOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::NullWireOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class NullWireOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  NullWireOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class NullWireOpGenericAdaptor : public detail::NullWireOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::NullWireOpGenericAdaptorBase;
public:
  NullWireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class NullWireOpAdaptor : public NullWireOpGenericAdaptor<::mlir::ValueRange> {
public:
  using NullWireOpGenericAdaptor::NullWireOpGenericAdaptor;
  NullWireOpAdaptor(NullWireOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class NullWireOp : public ::mlir::Op<NullWireOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::WireType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = NullWireOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = NullWireOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.null_wire");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::NullWireOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::RelaxSizeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RelaxSizeOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RelaxSizeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class RelaxSizeOpGenericAdaptor : public detail::RelaxSizeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RelaxSizeOpGenericAdaptorBase;
public:
  RelaxSizeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInputVec() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RelaxSizeOpAdaptor : public RelaxSizeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RelaxSizeOpGenericAdaptor::RelaxSizeOpGenericAdaptor;
  RelaxSizeOpAdaptor(RelaxSizeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RelaxSizeOp : public ::mlir::Op<RelaxSizeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::VeqType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RelaxSizeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RelaxSizeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.relax_size");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::VeqType> getInputVec();
  ::mlir::MutableOperandRange getInputVecMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value inputVec);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputVec);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::RelaxSizeOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::ResetOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ResetOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ResetOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ResetOpGenericAdaptor : public detail::ResetOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ResetOpGenericAdaptorBase;
public:
  ResetOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getTargets() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ResetOpAdaptor : public ResetOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ResetOpGenericAdaptor::ResetOpGenericAdaptor;
  ResetOpAdaptor(ResetOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ResetOp : public ::mlir::Op<ResetOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::cudaq::QuantumGate, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ResetOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ResetOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.reset");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getTargets();
  ::mlir::MutableOperandRange getTargetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getWires();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange wires, ::mlir::Value targets);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
  void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::
      EffectInstance<mlir::MemoryEffects::Effect>> &effects) {
    quake::getResetEffectsImpl(effects, getTargets());
  }
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::ResetOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::ReturnWireOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReturnWireOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ReturnWireOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ReturnWireOpGenericAdaptor : public detail::ReturnWireOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReturnWireOpGenericAdaptorBase;
public:
  ReturnWireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getTarget() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReturnWireOpAdaptor : public ReturnWireOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReturnWireOpGenericAdaptor::ReturnWireOpGenericAdaptor;
  ReturnWireOpAdaptor(ReturnWireOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReturnWireOp : public ::mlir::Op<ReturnWireOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReturnWireOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReturnWireOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.return_wire");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::WireType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value target);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::ReturnWireOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::SinkCableOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SinkCableOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SinkCableOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class SinkCableOpGenericAdaptor : public detail::SinkCableOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SinkCableOpGenericAdaptorBase;
public:
  SinkCableOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getCable() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SinkCableOpAdaptor : public SinkCableOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SinkCableOpGenericAdaptor::SinkCableOpGenericAdaptor;
  SinkCableOpAdaptor(SinkCableOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SinkCableOp : public ::mlir::Op<SinkCableOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SinkCableOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SinkCableOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.sink_cable");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::CableType> getCable();
  ::mlir::MutableOperandRange getCableMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value cable);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cable);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::SinkCableOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::SinkOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SinkOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SinkOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class SinkOpGenericAdaptor : public detail::SinkOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SinkOpGenericAdaptorBase;
public:
  SinkOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getTarget() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SinkOpAdaptor : public SinkOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SinkOpGenericAdaptor::SinkOpGenericAdaptor;
  SinkOpAdaptor(SinkOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SinkOp : public ::mlir::Op<SinkOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SinkOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SinkOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.sink");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::WireType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value target);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::SinkOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::SubVeqOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SubVeqOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SubVeqOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getRawLowerAttr();
  uint64_t getRawLower();
  ::mlir::IntegerAttr getRawUpperAttr();
  uint64_t getRawUpper();
};
} // namespace detail
template <typename RangeT>
class SubVeqOpGenericAdaptor : public detail::SubVeqOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SubVeqOpGenericAdaptorBase;
public:
  SubVeqOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVeq() {
    return *getODSOperands(0).begin();
  }

  ValueT getLower() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : *operands.begin();
  }

  ValueT getUpper() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : *operands.begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SubVeqOpAdaptor : public SubVeqOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SubVeqOpGenericAdaptor::SubVeqOpGenericAdaptor;
  SubVeqOpAdaptor(SubVeqOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SubVeqOp : public ::mlir::Op<SubVeqOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::VeqType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SubVeqOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SubVeqOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("rawLower"), ::llvm::StringRef("rawUpper")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getRawLowerAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getRawLowerAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getRawUpperAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getRawUpperAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.subveq");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::VeqType> getVeq();
  ::mlir::Value getLower();
  ::mlir::Value getUpper();
  ::mlir::MutableOperandRange getVeqMutable();
  ::mlir::MutableOperandRange getLowerMutable();
  ::mlir::MutableOperandRange getUpperMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::quake::VeqType> getQsub();
  ::mlir::IntegerAttr getRawLowerAttr();
  uint64_t getRawLower();
  ::mlir::IntegerAttr getRawUpperAttr();
  uint64_t getRawUpper();
  void setRawLowerAttr(::mlir::IntegerAttr attr);
  void setRawLower(uint64_t attrValue);
  void setRawUpperAttr(::mlir::IntegerAttr attr);
  void setRawUpper(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Type veqTy, mlir::Value input, mlir::Value lower, mlir::Value upper);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Type veqTy, mlir::Value input, std::int64_t lower, std::int64_t upper);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type qsub, ::mlir::Value veq, /*optional*/::mlir::Value lower, /*optional*/::mlir::Value upper, ::mlir::IntegerAttr rawLower, ::mlir::IntegerAttr rawUpper);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value veq, /*optional*/::mlir::Value lower, /*optional*/::mlir::Value upper, ::mlir::IntegerAttr rawLower, ::mlir::IntegerAttr rawUpper);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type qsub, ::mlir::Value veq, /*optional*/::mlir::Value lower, /*optional*/::mlir::Value upper, uint64_t rawLower, uint64_t rawUpper);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value veq, /*optional*/::mlir::Value lower, /*optional*/::mlir::Value upper, uint64_t rawLower, uint64_t rawUpper);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static constexpr std::size_t kDynamicIndex =
    std::numeric_limits<std::size_t>::max();

  bool hasConstantLowerBound() { return getRawLower() != kDynamicIndex; }
  bool hasConstantUpperBound() { return getRawUpper() != kDynamicIndex; }
  std::size_t getConstantLowerBound() { return getRawLower(); }
  std::size_t getConstantUpperBound() { return getRawUpper(); }
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::SubVeqOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::TerminateCableOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TerminateCableOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TerminateCableOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class TerminateCableOpGenericAdaptor : public detail::TerminateCableOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TerminateCableOpGenericAdaptorBase;
public:
  TerminateCableOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getCable() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TerminateCableOpAdaptor : public TerminateCableOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TerminateCableOpGenericAdaptor::TerminateCableOpGenericAdaptor;
  TerminateCableOpAdaptor(TerminateCableOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TerminateCableOp : public ::mlir::Op<TerminateCableOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TerminateCableOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TerminateCableOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.terminate_cable");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::CableType> getCable();
  ::mlir::MutableOperandRange getCableMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::mlir::Value cable);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::TerminateCableOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::ToControlOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ToControlOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ToControlOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ToControlOpGenericAdaptor : public detail::ToControlOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ToControlOpGenericAdaptorBase;
public:
  ToControlOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getQubit() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ToControlOpAdaptor : public ToControlOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ToControlOpGenericAdaptor::ToControlOpGenericAdaptor;
  ToControlOpAdaptor(ToControlOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ToControlOp : public ::mlir::Op<ToControlOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::ControlType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ToControlOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ToControlOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.to_ctrl");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::WireType> getQubit();
  ::mlir::MutableOperandRange getQubitMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value qubit);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value qubit);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::ToControlOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::UnwrapOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class UnwrapOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  UnwrapOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class UnwrapOpGenericAdaptor : public detail::UnwrapOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::UnwrapOpGenericAdaptorBase;
public:
  UnwrapOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRefValue() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class UnwrapOpAdaptor : public UnwrapOpGenericAdaptor<::mlir::ValueRange> {
public:
  using UnwrapOpGenericAdaptor::UnwrapOpGenericAdaptor;
  UnwrapOpAdaptor(UnwrapOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class UnwrapOp : public ::mlir::Op<UnwrapOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::quake::WireType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = UnwrapOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = UnwrapOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.unwrap");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::RefType> getRefValue();
  ::mlir::MutableOperandRange getRefValueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value ref_value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ref_value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::UnwrapOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::VeqSizeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class VeqSizeOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  VeqSizeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class VeqSizeOpGenericAdaptor : public detail::VeqSizeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::VeqSizeOpGenericAdaptorBase;
public:
  VeqSizeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVeq() {
    return *getODSOperands(0).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class VeqSizeOpAdaptor : public VeqSizeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using VeqSizeOpGenericAdaptor::VeqSizeOpGenericAdaptor;
  VeqSizeOpAdaptor(VeqSizeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class VeqSizeOp : public ::mlir::Op<VeqSizeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = VeqSizeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = VeqSizeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.veq_size");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::VeqType> getVeq();
  ::mlir::MutableOperandRange getVeqMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getSize();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type size, ::mlir::Value veq);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value veq);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::VeqSizeOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::WireSetOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class WireSetOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  WireSetOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::IntegerAttr getCardinalityAttr();
  uint32_t getCardinality();
  ::mlir::ElementsAttr getAdjacencyAttr();
  ::std::optional< ::mlir::ElementsAttr > getAdjacency();
};
} // namespace detail
template <typename RangeT>
class WireSetOpGenericAdaptor : public detail::WireSetOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WireSetOpGenericAdaptorBase;
public:
  WireSetOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WireSetOpAdaptor : public WireSetOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WireSetOpGenericAdaptor::WireSetOpGenericAdaptor;
  WireSetOpAdaptor(WireSetOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class WireSetOp : public ::mlir::Op<WireSetOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::IsIsolatedFromAbove, ::mlir::SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WireSetOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WireSetOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("adjacency"), ::llvm::StringRef("cardinality"), ::llvm::StringRef("sym_name")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAdjacencyAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAdjacencyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getCardinalityAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getCardinalityAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getSymNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getSymNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.wire_set");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::IntegerAttr getCardinalityAttr();
  uint32_t getCardinality();
  ::mlir::ElementsAttr getAdjacencyAttr();
  ::std::optional< ::mlir::ElementsAttr > getAdjacency();
  void setSymNameAttr(::mlir::StringAttr attr);
  void setSymName(::llvm::StringRef attrValue);
  void setCardinalityAttr(::mlir::IntegerAttr attr);
  void setCardinality(uint32_t attrValue);
  void setAdjacencyAttr(::mlir::ElementsAttr attr);
  ::mlir::Attribute removeAdjacencyAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr cardinality, /*optional*/::mlir::ElementsAttr adjacency);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr cardinality, /*optional*/::mlir::ElementsAttr adjacency);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, uint32_t cardinality, /*optional*/::mlir::ElementsAttr adjacency);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, uint32_t cardinality, /*optional*/::mlir::ElementsAttr adjacency);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::WireSetOp)

namespace quake {

//===----------------------------------------------------------------------===//
// ::quake::WrapOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class WrapOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  WrapOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class WrapOpGenericAdaptor : public detail::WrapOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WrapOpGenericAdaptorBase;
public:
  WrapOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getWireValue() {
    return *getODSOperands(0).begin();
  }

  ValueT getRefValue() {
    return *getODSOperands(1).begin();
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WrapOpAdaptor : public WrapOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WrapOpGenericAdaptor::WrapOpGenericAdaptor;
  WrapOpAdaptor(WrapOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class WrapOp : public ::mlir::Op<WrapOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WrapOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WrapOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("quake.wrap");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::quake::WireType> getWireValue();
  ::mlir::TypedValue<::quake::RefType> getRefValue();
  ::mlir::MutableOperandRange getWireValueMutable();
  ::mlir::MutableOperandRange getRefValueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value wire_value, ::mlir::Value ref_value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value wire_value, ::mlir::Value ref_value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace quake
MLIR_DECLARE_EXPLICIT_TYPE_ID(::quake::WrapOp)


#endif  // GET_OP_CLASSES
```