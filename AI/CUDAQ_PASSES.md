```C++
#ifdef GEN_PASS_REGISTRATION

//===----------------------------------------------------------------------===//
// AddMeasurements Registration
//===----------------------------------------------------------------------===//

inline void registerAddMeasurements() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createAddMeasurements();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerAddMeasurementsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createAddMeasurements();
  });
}

//===----------------------------------------------------------------------===//
// AddWireset Registration
//===----------------------------------------------------------------------===//

inline void registerAddWireset() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createAddWireset();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerAddWiresetPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createAddWireset();
  });
}

//===----------------------------------------------------------------------===//
// ApplyControlNegations Registration
//===----------------------------------------------------------------------===//

inline void registerApplyControlNegations() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createApplyControlNegations();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerApplyControlNegationsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createApplyControlNegations();
  });
}

//===----------------------------------------------------------------------===//
// ApplySpecialization Registration
//===----------------------------------------------------------------------===//

inline void registerApplySpecialization() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createApplySpecialization();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerApplySpecializationPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createApplySpecialization();
  });
}

//===----------------------------------------------------------------------===//
// ArgumentSynthesis Registration
//===----------------------------------------------------------------------===//

inline void registerArgumentSynthesis() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createArgumentSynthesis();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerArgumentSynthesisPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createArgumentSynthesis();
  });
}

//===----------------------------------------------------------------------===//
// AssignWireIndices Registration
//===----------------------------------------------------------------------===//

inline void registerAssignWireIndices() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createAssignWireIndices();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerAssignWireIndicesPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createAssignWireIndices();
  });
}

//===----------------------------------------------------------------------===//
// BasisConversionPass Registration
//===----------------------------------------------------------------------===//

inline void registerBasisConversionPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createBasisConversionPass();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerBasisConversionPassPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createBasisConversionPass();
  });
}

//===----------------------------------------------------------------------===//
// CheckKernelCalls Registration
//===----------------------------------------------------------------------===//

inline void registerCheckKernelCalls() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createCheckKernelCalls();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerCheckKernelCallsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createCheckKernelCalls();
  });
}

//===----------------------------------------------------------------------===//
// ClassicalOptimization Registration
//===----------------------------------------------------------------------===//

inline void registerClassicalOptimization() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createClassicalOptimization();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerClassicalOptimizationPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createClassicalOptimization();
  });
}

//===----------------------------------------------------------------------===//
// CombineMeasurements Registration
//===----------------------------------------------------------------------===//

inline void registerCombineMeasurements() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createCombineMeasurements();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerCombineMeasurementsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createCombineMeasurements();
  });
}

//===----------------------------------------------------------------------===//
// CombineQuantumAllocations Registration
//===----------------------------------------------------------------------===//

inline void registerCombineQuantumAllocations() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createCombineQuantumAllocations();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerCombineQuantumAllocationsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createCombineQuantumAllocations();
  });
}

//===----------------------------------------------------------------------===//
// ConstantPropagation Registration
//===----------------------------------------------------------------------===//

inline void registerConstantPropagation() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createConstantPropagation();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerConstantPropagationPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createConstantPropagation();
  });
}

//===----------------------------------------------------------------------===//
// ConvertToCFG Registration
//===----------------------------------------------------------------------===//

inline void registerConvertToCFG() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createConvertToCFG();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerConvertToCFGPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createConvertToCFG();
  });
}

//===----------------------------------------------------------------------===//
// ConvertToCFGPrep Registration
//===----------------------------------------------------------------------===//

inline void registerConvertToCFGPrep() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createConvertToCFGPrep();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerConvertToCFGPrepPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createConvertToCFGPrep();
  });
}

//===----------------------------------------------------------------------===//
// ConvertToDirectCalls Registration
//===----------------------------------------------------------------------===//

inline void registerConvertToDirectCalls() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createConvertToDirectCalls();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerConvertToDirectCallsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createConvertToDirectCalls();
  });
}

//===----------------------------------------------------------------------===//
// DeadStoreRemoval Registration
//===----------------------------------------------------------------------===//

inline void registerDeadStoreRemoval() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createDeadStoreRemoval();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerDeadStoreRemovalPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createDeadStoreRemoval();
  });
}

//===----------------------------------------------------------------------===//
// DecompositionPass Registration
//===----------------------------------------------------------------------===//

inline void registerDecompositionPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createDecompositionPass();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerDecompositionPassPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createDecompositionPass();
  });
}

//===----------------------------------------------------------------------===//
// DelayMeasurements Registration
//===----------------------------------------------------------------------===//

inline void registerDelayMeasurements() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createDelayMeasurementsPass();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerDelayMeasurementsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createDelayMeasurementsPass();
  });
}

//===----------------------------------------------------------------------===//
// DeleteStates Registration
//===----------------------------------------------------------------------===//

inline void registerDeleteStates() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createDeleteStates();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerDeleteStatesPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createDeleteStates();
  });
}

//===----------------------------------------------------------------------===//
// DependencyAnalysis Registration
//===----------------------------------------------------------------------===//

inline void registerDependencyAnalysis() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createDependencyAnalysis();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerDependencyAnalysisPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createDependencyAnalysis();
  });
}

//===----------------------------------------------------------------------===//
// DistributedDeviceCall Registration
//===----------------------------------------------------------------------===//

inline void registerDistributedDeviceCall() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createDistributedDeviceCall();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerDistributedDeviceCallPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createDistributedDeviceCall();
  });
}

//===----------------------------------------------------------------------===//
// EraseNoise Registration
//===----------------------------------------------------------------------===//

inline void registerEraseNoise() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createEraseNoise();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerEraseNoisePass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createEraseNoise();
  });
}

//===----------------------------------------------------------------------===//
// EraseNopCalls Registration
//===----------------------------------------------------------------------===//

inline void registerEraseNopCalls() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createEraseNopCalls();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerEraseNopCallsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createEraseNopCalls();
  });
}

//===----------------------------------------------------------------------===//
// EraseVectorCopyCtor Registration
//===----------------------------------------------------------------------===//

inline void registerEraseVectorCopyCtor() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createEraseVectorCopyCtor();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerEraseVectorCopyCtorPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createEraseVectorCopyCtor();
  });
}

//===----------------------------------------------------------------------===//
// ExpandControlVeqs Registration
//===----------------------------------------------------------------------===//

inline void registerExpandControlVeqs() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createExpandControlVeqs();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerExpandControlVeqsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createExpandControlVeqs();
  });
}

//===----------------------------------------------------------------------===//
// ExpandMeasurements Registration
//===----------------------------------------------------------------------===//

inline void registerExpandMeasurements() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createExpandMeasurementsPass();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerExpandMeasurementsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createExpandMeasurementsPass();
  });
}

//===----------------------------------------------------------------------===//
// FactorQuantumAllocations Registration
//===----------------------------------------------------------------------===//

inline void registerFactorQuantumAllocations() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createFactorQuantumAllocations();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerFactorQuantumAllocationsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createFactorQuantumAllocations();
  });
}

//===----------------------------------------------------------------------===//
// GenerateDeviceCodeLoader Registration
//===----------------------------------------------------------------------===//

inline void registerGenerateDeviceCodeLoader() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createGenerateDeviceCodeLoader();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerGenerateDeviceCodeLoaderPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createGenerateDeviceCodeLoader();
  });
}

//===----------------------------------------------------------------------===//
// GenerateKernelExecution Registration
//===----------------------------------------------------------------------===//

inline void registerGenerateKernelExecution() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createGenerateKernelExecution();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerGenerateKernelExecutionPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createGenerateKernelExecution();
  });
}

//===----------------------------------------------------------------------===//
// GetConcreteMatrix Registration
//===----------------------------------------------------------------------===//

inline void registerGetConcreteMatrix() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createGetConcreteMatrix();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerGetConcreteMatrixPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createGetConcreteMatrix();
  });
}

//===----------------------------------------------------------------------===//
// GlobalizeArrayValues Registration
//===----------------------------------------------------------------------===//

inline void registerGlobalizeArrayValues() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createGlobalizeArrayValues();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerGlobalizeArrayValuesPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createGlobalizeArrayValues();
  });
}

//===----------------------------------------------------------------------===//
// LambdaLifting Registration
//===----------------------------------------------------------------------===//

inline void registerLambdaLifting() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createLambdaLiftingPass();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerLambdaLiftingPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createLambdaLiftingPass();
  });
}

//===----------------------------------------------------------------------===//
// LiftArrayAlloc Registration
//===----------------------------------------------------------------------===//

inline void registerLiftArrayAlloc() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLiftArrayAlloc();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerLiftArrayAllocPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLiftArrayAlloc();
  });
}

//===----------------------------------------------------------------------===//
// LinearCtrlRelations Registration
//===----------------------------------------------------------------------===//

inline void registerLinearCtrlRelations() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLinearCtrlRelations();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerLinearCtrlRelationsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLinearCtrlRelations();
  });
}

//===----------------------------------------------------------------------===//
// LoopNormalize Registration
//===----------------------------------------------------------------------===//

inline void registerLoopNormalize() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLoopNormalize();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerLoopNormalizePass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLoopNormalize();
  });
}

//===----------------------------------------------------------------------===//
// LoopPeeling Registration
//===----------------------------------------------------------------------===//

inline void registerLoopPeeling() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLoopPeeling();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerLoopPeelingPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLoopPeeling();
  });
}

//===----------------------------------------------------------------------===//
// LoopUnroll Registration
//===----------------------------------------------------------------------===//

inline void registerLoopUnroll() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLoopUnroll();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerLoopUnrollPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLoopUnroll();
  });
}

//===----------------------------------------------------------------------===//
// MappingFunc Registration
//===----------------------------------------------------------------------===//

inline void registerMappingFunc() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createMappingFunc();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerMappingFuncPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createMappingFunc();
  });
}

//===----------------------------------------------------------------------===//
// MappingPrep Registration
//===----------------------------------------------------------------------===//

inline void registerMappingPrep() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createMappingPrep();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerMappingPrepPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createMappingPrep();
  });
}

//===----------------------------------------------------------------------===//
// MemToReg Registration
//===----------------------------------------------------------------------===//

inline void registerMemToReg() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createMemToReg();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerMemToRegPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createMemToReg();
  });
}

//===----------------------------------------------------------------------===//
// MultiControlDecompositionPass Registration
//===----------------------------------------------------------------------===//

inline void registerMultiControlDecompositionPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createMultiControlDecompositionPass();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerMultiControlDecompositionPassPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createMultiControlDecompositionPass();
  });
}

//===----------------------------------------------------------------------===//
// ObserveAnsatz Registration
//===----------------------------------------------------------------------===//

inline void registerObserveAnsatz() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createObserveAnsatz();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerObserveAnsatzPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createObserveAnsatz();
  });
}

//===----------------------------------------------------------------------===//
// PromoteRefToVeqAlloc Registration
//===----------------------------------------------------------------------===//

inline void registerPromoteRefToVeqAlloc() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createPromoteRefToVeqAlloc();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerPromoteRefToVeqAllocPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createPromoteRefToVeqAlloc();
  });
}

//===----------------------------------------------------------------------===//
// PruneCtrlRelations Registration
//===----------------------------------------------------------------------===//

inline void registerPruneCtrlRelations() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createPruneCtrlRelations();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerPruneCtrlRelationsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createPruneCtrlRelations();
  });
}

//===----------------------------------------------------------------------===//
// PySynthCallableBlockArgs Registration
//===----------------------------------------------------------------------===//

inline void registerPySynthCallableBlockArgs() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createPySynthCallableBlockArgs();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerPySynthCallableBlockArgsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createPySynthCallableBlockArgs();
  });
}

//===----------------------------------------------------------------------===//
// QuakeAddDeallocs Registration
//===----------------------------------------------------------------------===//

inline void registerQuakeAddDeallocs() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createQuakeAddDeallocs();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerQuakeAddDeallocsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createQuakeAddDeallocs();
  });
}

//===----------------------------------------------------------------------===//
// QuakeAddMetadata Registration
//===----------------------------------------------------------------------===//

inline void registerQuakeAddMetadata() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createQuakeAddMetadata();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerQuakeAddMetadataPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createQuakeAddMetadata();
  });
}

//===----------------------------------------------------------------------===//
// QuakePropagateMetadata Registration
//===----------------------------------------------------------------------===//

inline void registerQuakePropagateMetadata() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createQuakePropagateMetadata();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerQuakePropagateMetadataPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createQuakePropagateMetadata();
  });
}

//===----------------------------------------------------------------------===//
// QuakeSimplify Registration
//===----------------------------------------------------------------------===//

inline void registerQuakeSimplify() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createQuakeSimplify();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerQuakeSimplifyPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createQuakeSimplify();
  });
}

//===----------------------------------------------------------------------===//
// QuakeSynthesize Registration
//===----------------------------------------------------------------------===//

inline void registerQuakeSynthesize() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createQuakeSynthesizer();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerQuakeSynthesizePass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return cudaq::opt::createQuakeSynthesizer();
  });
}

//===----------------------------------------------------------------------===//
// RegToMem Registration
//===----------------------------------------------------------------------===//

inline void registerRegToMem() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createRegToMem();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerRegToMemPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createRegToMem();
  });
}

//===----------------------------------------------------------------------===//
// ReplaceStateWithKernel Registration
//===----------------------------------------------------------------------===//

inline void registerReplaceStateWithKernel() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createReplaceStateWithKernel();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerReplaceStateWithKernelPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createReplaceStateWithKernel();
  });
}

//===----------------------------------------------------------------------===//
// ResourceCountPreprocess Registration
//===----------------------------------------------------------------------===//

inline void registerResourceCountPreprocess() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createResourceCountPreprocess();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerResourceCountPreprocessPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createResourceCountPreprocess();
  });
}

//===----------------------------------------------------------------------===//
// SROA Registration
//===----------------------------------------------------------------------===//

inline void registerSROA() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createSROA();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerSROAPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createSROA();
  });
}

//===----------------------------------------------------------------------===//
// StackFramePrealloc Registration
//===----------------------------------------------------------------------===//

inline void registerStackFramePrealloc() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createStackFramePrealloc();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerStackFramePreallocPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createStackFramePrealloc();
  });
}

//===----------------------------------------------------------------------===//
// StatePreparation Registration
//===----------------------------------------------------------------------===//

inline void registerStatePreparation() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createStatePreparation();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerStatePreparationPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createStatePreparation();
  });
}

//===----------------------------------------------------------------------===//
// UnitarySynthesis Registration
//===----------------------------------------------------------------------===//

inline void registerUnitarySynthesis() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createUnitarySynthesis();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerUnitarySynthesisPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createUnitarySynthesis();
  });
}

//===----------------------------------------------------------------------===//
// UnwindLowering Registration
//===----------------------------------------------------------------------===//

inline void registerUnwindLowering() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createUnwindLowering();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerUnwindLoweringPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createUnwindLowering();
  });
}

//===----------------------------------------------------------------------===//
// UpdateRegisterNames Registration
//===----------------------------------------------------------------------===//

inline void registerUpdateRegisterNames() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createUpdateRegisterNames();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerUpdateRegisterNamesPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createUpdateRegisterNames();
  });
}

//===----------------------------------------------------------------------===//
// VariableCoalesce Registration
//===----------------------------------------------------------------------===//

inline void registerVariableCoalesce() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createVariableCoalesce();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerVariableCoalescePass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createVariableCoalesce();
  });
}

//===----------------------------------------------------------------------===//
// WriteAfterWriteElimination Registration
//===----------------------------------------------------------------------===//

inline void registerWriteAfterWriteElimination() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createWriteAfterWriteElimination();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerWriteAfterWriteEliminationPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createWriteAfterWriteElimination();
  });
}

//===----------------------------------------------------------------------===//
// OptTransforms Registration
//===----------------------------------------------------------------------===//

inline void registerOptTransformsPasses() {
  registerAddMeasurements();
  registerAddWireset();
  registerApplyControlNegations();
  registerApplySpecialization();
  registerArgumentSynthesis();
  registerAssignWireIndices();
  registerBasisConversionPass();
  registerCheckKernelCalls();
  registerClassicalOptimization();
  registerCombineMeasurements();
  registerCombineQuantumAllocations();
  registerConstantPropagation();
  registerConvertToCFG();
  registerConvertToCFGPrep();
  registerConvertToDirectCalls();
  registerDeadStoreRemoval();
  registerDecompositionPass();
  registerDelayMeasurements();
  registerDeleteStates();
  registerDependencyAnalysis();
  registerDistributedDeviceCall();
  registerEraseNoise();
  registerEraseNopCalls();
  registerEraseVectorCopyCtor();
  registerExpandControlVeqs();
  registerExpandMeasurements();
  registerFactorQuantumAllocations();
  registerGenerateDeviceCodeLoader();
  registerGenerateKernelExecution();
  registerGetConcreteMatrix();
  registerGlobalizeArrayValues();
  registerLambdaLifting();
  registerLiftArrayAlloc();
  registerLinearCtrlRelations();
  registerLoopNormalize();
  registerLoopPeeling();
  registerLoopUnroll();
  registerMappingFunc();
  registerMappingPrep();
  registerMemToReg();
  registerMultiControlDecompositionPass();
  registerObserveAnsatz();
  registerPromoteRefToVeqAlloc();
  registerPruneCtrlRelations();
  registerPySynthCallableBlockArgs();
  registerQuakeAddDeallocs();
  registerQuakeAddMetadata();
  registerQuakePropagateMetadata();
  registerQuakeSimplify();
  registerQuakeSynthesize();
  registerRegToMem();
  registerReplaceStateWithKernel();
  registerResourceCountPreprocess();
  registerSROA();
  registerStackFramePrealloc();
  registerStatePreparation();
  registerUnitarySynthesis();
  registerUnwindLowering();
  registerUpdateRegisterNames();
  registerVariableCoalesce();
  registerWriteAfterWriteElimination();
}
#undef GEN_PASS_REGISTRATION
#endif // GEN_PASS_REGISTRATION
// Deprecated. Please use the new per-pass macros.
#ifdef GEN_PASS_CLASSES
```