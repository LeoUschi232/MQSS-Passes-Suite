/* This code and any associated documentation is provided "as is"
Copyright 2025 Munich Quantum Software Stack Project
Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://github.com/Munich-Quantum-Software-Stack/passes/blob/develop/LICENSE
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------------------
  author Martin Letras
  date   February 2025
  version 1.0
  brief
    Definition of map used to insert quantum gates into a MLIR module. It
receives as input a tag that identifies the quantum gate, the list of arguments,
control and target qubits.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "Interfaces/QASMToQuake.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mqss::support::quakeDialect;

/*
Suggested Missing Gates (Only Suggest, Do Not Implement)
Based on common QASM/OpenQASM3 gates and Quake dialect:
    xxminusyy, xxplusyy: For variational circuits (e.g., QAOA).
    rzx: Already discussed, but full impl.
    c3x, c4x: Higher controlled-X for multi-control Toffoli variants.
    ms: Mølmer–Sørensen gate for ion traps.
    fswap: Fermionic SWAP.
    givens: Givens rotation for chemistry sims.
    r1: Arbitrary phase on |1> (as in Quake).
    barrier: For compilation hints (no-op).
 */

void mqss::interfaces::insertQASMGateIntoQuakeModule(
    std::string gateId, OpBuilder &builder, Location loc,
    std::vector<mlir::Value> vecParams, std::vector<mlir::Value> vecControls,
    std::vector<mlir::Value> vecTargets, bool adj) {
  std::transform(gateId.begin(), gateId.end(), gateId.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  mlir::ValueRange params(vecParams);
  mlir::ValueRange controls(vecControls);
  mlir::ValueRange targets(vecTargets);
  assert(targets.empty() && "ill-formed gate");
  ValueRange empty;
  mlir::Value plusHalfPi = createFloatValue(builder, loc, PI_2);
  mlir::Value plusQuarterPi = createFloatValue(builder, loc, PI_4);
  mlir::Value minusHalfPi = createFloatValue(builder, loc, -PI_2);
  mlir::Value minusQuarterPi = createFloatValue(builder, loc, -PI_4);
#ifdef DEBUG
  std::cout << "gate " << gateId << std::endl;
  std::cout << "controls size " << controls.size() << std::endl;
  std::cout << "target size " << targets.size() << std::endl;
  std::cout << "params size " << params.size() << std::endl;
#endif
  static const std::unordered_map<std::string, std::function<void()>> gateMap =
      {{"id", [&]() { /* Identity does nothing. */ }},
       {"gphase", [&]() { /* Global phase does not affect measurement. */ }},
       {"x",
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 "ill-formed x gate");
          builder.create<quake::XOp>(loc, false, empty, empty, targets);
        }},
       {"y",
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 "ill-formed y gate");
          builder.create<quake::YOp>(loc, false, empty, empty, targets);
        }},
       {"z",
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 "ill-formed z gate");
          builder.create<quake::ZOp>(loc, false, empty, empty, targets);
        }},
       {"h",
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 "ill-formed h gate");
          builder.create<quake::HOp>(loc, false, empty, empty, targets);
        }},
       {"s",
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 !adj && "ill-formed s gate");
          builder.create<quake::SOp>(loc, false, empty, empty, targets);
        }},
       {"sdg",
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 adj && "ill-formed sdg gate");
          builder.create<quake::SOp>(loc, true, empty, empty, targets);
        }},
       {"t",
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 !adj && "ill-formed t gate");
          builder.create<quake::TOp>(loc, false, empty, empty, targets);
        }},
       {"tdg",
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 adj && "ill-formed tdg gate");
          builder.create<quake::TOp>(loc, true, empty, empty, targets);
        }},
       {"rx",
        [&]() {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed rx gate");
          builder.create<quake::RxOp>(loc, false, params, empty, targets);
        }},
       {"ry",
        [&]() {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed ry gate");
          builder.create<quake::RyOp>(loc, false, params, empty, targets);
        }},
       {"rz",
        [&]() {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed rz gate");
          builder.create<quake::RzOp>(loc, false, params, empty, targets);
        }},
       {"p",
        // P(θ) = exp(iθ/2)*Rz(θ)
        // The global phase factor is irrelevant for the final measurement
        // result in quantum computing because it does not affect the
        // probabilities of measurement outcomes.
        // Therefore instead of the phase gate one can just use the Rz gate.
        [&]() {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed p gate");
          builder.create<quake::RzOp>(loc, false, params, empty, targets);
        }},
       {"phase",
        [&]() {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed phase gate");
          builder.create<quake::RzOp>(loc, false, params, empty, targets);
        }},
       {"sx",
        // Sx = Rx(π/2)
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 !adj && "ill-formed sx gate");
          builder.create<quake::RxOp>(loc, false, plusHalfPi, empty, targets);
        }},
       {"sxdg",
        // Sxdg = Rx(-π/2)
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 adj && "ill-formed sxdg gate");
          builder.create<quake::RxOp>(loc, false, minusHalfPi, empty, targets);
        }},
       {"u1",
        // U1 gate is the same thing as the Phase gate.
        // U1(θ) = exp(iθ/2)*Rz(θ)
        [&]() {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed u1 gate");
          builder.create<quake::RzOp>(loc, false, params, empty, targets);
        }},
       {"u2",
        // U2(ϕ,λ) = exp(i*(ϕ-λ)/2)*Rz(ϕ)Ry(π/2)Rz(λ)
        [&]() {
          assert(params.size() == 2 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed u2 gate");
          double phi = extractDoubleArgumentValue(params[0].getDefiningOp());
          double lambda = extractDoubleArgumentValue(params[1].getDefiningOp());
          mlir::Value param1 = createFloatValue(builder, loc, lambda);
          mlir::Value param2 = plusHalfPi;
          mlir::Value param3 = createFloatValue(builder, loc, phi);

          builder.create<quake::RzOp>(loc, false, param1, empty, targets);
          builder.create<quake::RyOp>(loc, false, param2, empty, targets);
          builder.create<quake::RzOp>(loc, false, param3, empty, targets);
        }},
       {"u3",
        // U3(θ,ϕ,λ) = Rz(ϕ)Ry(θ)Rz(λ)
        [&]() {
          assert(params.size() == 3 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed u3 gate");
          double theta = extractDoubleArgumentValue(params[0].getDefiningOp());
          double phi = extractDoubleArgumentValue(params[1].getDefiningOp());
          double lambda = extractDoubleArgumentValue(params[2].getDefiningOp());
          mlir::Value param1 = createFloatValue(builder, loc, lambda);
          mlir::Value param2 = createFloatValue(builder, loc, theta);
          mlir::Value param3 = createFloatValue(builder, loc, phi);

          builder.create<quake::RzOp>(loc, false, param1, empty, targets);
          builder.create<quake::RyOp>(loc, false, param2, empty, targets);
          builder.create<quake::RzOp>(loc, false, param3, empty, targets);
        }},
       {"u",
        // U(θ,ϕ,λ) = Rz(ϕ)Ry(θ)Rz(λ)
        [&]() {
          assert(params.size() == 3 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed u gate");
          double theta = extractDoubleArgumentValue(params[0].getDefiningOp());
          double phi = extractDoubleArgumentValue(params[1].getDefiningOp());
          double lambda = extractDoubleArgumentValue(params[2].getDefiningOp());
          mlir::Value param1 = createFloatValue(builder, loc, lambda);
          mlir::Value param2 = createFloatValue(builder, loc, theta);
          mlir::Value param3 = createFloatValue(builder, loc, phi);

          builder.create<quake::RzOp>(loc, false, param1, empty, targets);
          builder.create<quake::RyOp>(loc, false, param2, empty, targets);
          builder.create<quake::RzOp>(loc, false, param3, empty, targets);
        }},
       {"cx",
        [&]() {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 1 && "ill-formed cx gate");
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
        }},
       {"cy",
        [&]() {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 1 && "ill-formed cy gate");
          builder.create<quake::YOp>(loc, false, empty, controls, targets);
        }},
       {"cz",
        [&]() {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 1 && "ill-formed cz gate");
          builder.create<quake::ZOp>(loc, false, empty, controls, targets);
        }},
       {"ch",
        [&]() {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 1 && "ill-formed ch gate");
          builder.create<quake::HOp>(loc, false, empty, controls, targets);
        }},
       {"cs",
        [&]() {
          assert(params.empty() && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cs gate");
          builder.create<quake::SOp>(loc, false, empty, controls, targets);
        }},
       {"csdg",
        [&]() {
          assert(params.empty() && controls.size() == 1 && adj &&
                 targets.size() == 1 && "ill-formed csdg gate");
          builder.create<quake::SOp>(loc, true, empty, controls, targets);
        }},
       {"ct",
        [&]() {
          assert(params.empty() && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed ct gate");
          builder.create<quake::TOp>(loc, false, empty, controls, targets);
        }},
       {"ctdg",
        [&]() {
          assert(params.empty() && controls.size() == 1 && adj &&
                 targets.size() == 1 && "ill-formed ctdg gate");
          builder.create<quake::TOp>(loc, true, empty, controls, targets);
        }},
       {"crx",
        [&]() {
          assert(params.size() == 1 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed crx gate");
          builder.create<quake::RxOp>(loc, false, params, controls, targets);
        }},
       {"cry",
        [&]() {
          assert(params.size() == 1 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cry gate");
          builder.create<quake::RyOp>(loc, false, params, controls, targets);
        }},
       {"crz",
        [&]() {
          assert(params.size() == 1 && controls.size() == 1 &&
                 targets.size() == 1 && "ill-formed crz gate");
          builder.create<quake::RzOp>(loc, false, params, controls, targets);
        }},
       {"cp",
        // Cp(θ) q1, q2:
        //  Gphase(θ/4)
        //  Rz(θ/2) q1
        //  Cx q1, q2
        //  Rz(-θ/2) q2
        //  Cx q1, q2
        //  Rz(θ/2) q2
        [&]() {
          assert(params.size() == 1 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cp gate");
          double theta = extractDoubleArgumentValue(params[0].getDefiningOp());
          mlir::Value param1 = createFloatValue(builder, loc, 0.5 * theta);
          mlir::Value param2 = createFloatValue(builder, loc, -0.5 * theta);
          mlir::Value param3 = createFloatValue(builder, loc, 0.5 * theta);

          builder.create<quake::RzOp>(loc, false, param1, empty, controls);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RzOp>(loc, false, param2, empty, targets);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RzOp>(loc, false, param3, empty, targets);
        }},
       {"cphase",
        // Cphase(θ) q1, q2:
        //  Gphase(θ/4)
        //  Rz(θ/2) q1
        //  Cx q1, q2
        //  Rz(-θ/2) q2
        //  Cx q1, q2
        //  Rz(θ/2) q2
        [&]() {
          assert(params.size() == 1 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cphase gate");
          double theta = extractDoubleArgumentValue(params[0].getDefiningOp());
          mlir::Value param1 = createFloatValue(builder, loc, 0.5 * theta);
          mlir::Value param2 = createFloatValue(builder, loc, -0.5 * theta);
          mlir::Value param3 = createFloatValue(builder, loc, 0.5 * theta);

          builder.create<quake::RzOp>(loc, false, param1, empty, controls);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RzOp>(loc, false, param2, empty, targets);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RzOp>(loc, false, param3, empty, targets);
        }},
       {"cu1",
        // Cu1(θ) q1, q2:
        //  Gphase(θ/4)
        //  Rz(θ/2) q1
        //  Cx q1, q2
        //  Rz(-θ/2) q2
        //  Cx q1, q2
        //  Rz(θ/2) q2
        [&]() {
          assert(params.size() == 1 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cu1 gate");
          double theta = extractDoubleArgumentValue(params[0].getDefiningOp());
          mlir::Value param1 = createFloatValue(builder, loc, 0.5 * theta);
          mlir::Value param2 = createFloatValue(builder, loc, -0.5 * theta);
          mlir::Value param3 = createFloatValue(builder, loc, 0.5 * theta);

          builder.create<quake::RzOp>(loc, false, param1, empty, controls);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RzOp>(loc, false, param2, empty, targets);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RzOp>(loc, false, param3, empty, targets);
        }},
       {"cu2",
        // Cu2(ϕ,λ) q1, q2:
        //  Gphase((λ+φ)/4)
        //  Rz((λ+φ)/2) q1
        //  Rz((λ-φ)/2) q2
        //  Cx q1, q2
        //  Rz(-(λ+φ)/2) q2
        //  Ry(-π/4) q2
        //  Cx q1, q2
        //  Ry(π/4) q2
        //  Rz(φ) q2
        [&]() {
          assert(params.size() == 2 && controls.size() == 1 &&
                 targets.size() == 1 && "ill-formed cu2 gate");
          double phi = extractDoubleArgumentValue(params[0].getDefiningOp());
          double lambda = extractDoubleArgumentValue(params[1].getDefiningOp());
          mlir::Value param1 =
              createFloatValue(builder, loc, 0.5 * (lambda + phi));
          mlir::Value param2 =
              createFloatValue(builder, loc, 0.5 * (lambda - phi));
          mlir::Value param3 =
              createFloatValue(builder, loc, -0.5 * (lambda + phi));
          mlir::Value param4 = minusQuarterPi;
          mlir::Value param5 = plusQuarterPi;
          mlir::Value param6 = createFloatValue(builder, loc, phi);

          builder.create<quake::RzOp>(loc, false, param1, empty, controls);
          builder.create<quake::RzOp>(loc, false, param2, empty, targets);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RzOp>(loc, false, param3, empty, targets);
          builder.create<quake::RyOp>(loc, false, param4, empty, targets);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RyOp>(loc, false, param5, empty, targets);
          builder.create<quake::RzOp>(loc, false, param6, empty, targets);
        }},
       {"cu3",
        // Cu3(θ,ϕ,λ) q1, q2:
        //  Gphase((λ+ϕ)/4)
        //  Rz((λ+ϕ)/2) q1
        //  Rz((λ-ϕ)/2) q2
        //  Cx q1, q2
        //  Rz(-(λ+ϕ)/2) q2
        //  Ry(-θ/2) q2
        //  Cx q1, q2
        //  Ry(θ/2) q2
        //  rz(φ) q2
        [&]() {
          assert(params.size() == 3 && controls.size() == 1 &&
                 targets.size() == 1 && "ill-formed cu3 gate");
          double theta = extractDoubleArgumentValue(params[0].getDefiningOp());
          double phi = extractDoubleArgumentValue(params[1].getDefiningOp());
          double lambda = extractDoubleArgumentValue(params[2].getDefiningOp());
          mlir::Value param1 =
              createFloatValue(builder, loc, 0.5 * (lambda + phi));
          mlir::Value param2 =
              createFloatValue(builder, loc, 0.5 * (lambda - phi));
          mlir::Value param3 =
              createFloatValue(builder, loc, -0.5 * (lambda + phi));
          mlir::Value param4 = createFloatValue(builder, loc, -0.5 * theta);
          mlir::Value param5 = createFloatValue(builder, loc, 0.5 * theta);
          mlir::Value param6 = createFloatValue(builder, loc, phi);

          builder.create<quake::RzOp>(loc, false, param1, empty, controls);
          builder.create<quake::RzOp>(loc, false, param2, empty, targets);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RzOp>(loc, false, param3, empty, targets);
          builder.create<quake::RyOp>(loc, false, param4, empty, targets);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RyOp>(loc, false, param5, empty, targets);
          builder.create<quake::RzOp>(loc, false, param6, empty, targets);
        }},
       {"cu",
        // Cu(θ,ϕ,λ) q1, q2:
        //  Gphase((λ+ϕ)/4)
        //  Rz((λ+ϕ)/2) q1
        //  Rz((λ-ϕ)/2) q2
        //  Cx q1, q2
        //  Rz(-(λ+ϕ)/2) q2
        //  Ry(-θ/2) q2
        //  Cx q1, q2
        //  Ry(θ/2) q2
        //  rz(φ) q2
        [&]() {
          assert(params.size() == 3 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cu gate");
          double theta = extractDoubleArgumentValue(params[0].getDefiningOp());
          double phi = extractDoubleArgumentValue(params[1].getDefiningOp());
          double lambda = extractDoubleArgumentValue(params[2].getDefiningOp());
          mlir::Value param1 =
              createFloatValue(builder, loc, 0.5 * (lambda + phi));
          mlir::Value param2 =
              createFloatValue(builder, loc, 0.5 * (lambda - phi));
          mlir::Value param3 =
              createFloatValue(builder, loc, -0.5 * (lambda + phi));
          mlir::Value param4 = createFloatValue(builder, loc, -0.5 * theta);
          mlir::Value param5 = createFloatValue(builder, loc, 0.5 * theta);
          mlir::Value param6 = createFloatValue(builder, loc, phi);

          builder.create<quake::RzOp>(loc, false, param1, empty, controls);
          builder.create<quake::RzOp>(loc, false, param2, empty, targets);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RzOp>(loc, false, param3, empty, targets);
          builder.create<quake::RyOp>(loc, false, param4, empty, targets);
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
          builder.create<quake::RyOp>(loc, false, param5, empty, targets);
          builder.create<quake::RzOp>(loc, false, param6, empty, targets);
        }},
       {"swap",
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 2 &&
                 "ill-formed swap gate");
          builder.create<quake::SwapOp>(loc, false, params, controls, targets);
        }},
       {"iswap",
        // iSWAP q1, q2:
        //  S q1
        //  S q2
        //  H q1
        //  Cx q1, q2
        //  Cx q2, q1
        //  H q2
        [&]() {
          assert(params.empty() && controls.empty() && !adj &&
                 targets.size() == 2 && "ill-formed iswap gate");
          auto q1 = targets[0];
          auto q2 = targets[1];

          builder.create<quake::SOp>(loc, false, empty, empty, q1);
          builder.create<quake::SOp>(loc, false, empty, empty, q2);
          builder.create<quake::HOp>(loc, false, empty, empty, q1);
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
          builder.create<quake::XOp>(loc, false, empty, q2, q1);
          builder.create<quake::HOp>(loc, false, empty, empty, q2);
        }},
       {"iswapdg",
        [&]() {
          // iSWAPdg (q1, q2) {
          //  H q2
          //  Cx q2, q1
          //  Cx q1, q2
          //  H q1
          //  Sdg q1
          //  Sdg q2
          assert(params.empty() && controls.empty() && adj &&
                 targets.size() == 2 && "ill-formed iswapdg gate");
          auto q1 = targets[0];
          auto q2 = targets[1];

          builder.create<quake::HOp>(loc, false, empty, empty, q2);
          builder.create<quake::XOp>(loc, false, empty, q2, q1);
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
          builder.create<quake::HOp>(loc, false, empty, empty, q1);
          builder.create<quake::SOp>(loc, true, empty, empty, q1);
          builder.create<quake::SOp>(loc, true, empty, empty, q2);
        }},
       {"ccx",
        [&]() {
          assert(params.empty() && controls.size() == 2 &&
                 targets.size() == 1 && "ill-formed ccx gate");
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
        }},
       {"ccy",
        [&]() {
          assert(params.empty() && controls.size() == 2 &&
                 targets.size() == 1 && "ill-formed ccy gate");
          builder.create<quake::YOp>(loc, false, empty, controls, targets);
        }},
       {"ccz",
        [&]() {
          assert(params.empty() && controls.size() == 2 &&
                 targets.size() == 1 && "ill-formed ccz gate");
          builder.create<quake::ZOp>(loc, false, empty, controls, targets);
        }},
       {"cswap",
        [&]() {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 2 && "ill-formed cswap gate");
          builder.create<quake::SwapOp>(loc, false, empty, controls, targets);
        }},
       {"toffoli",
        [&]() {
          assert(params.empty() && controls.size() == 2 &&
                 targets.size() == 1 && "ill-formed toffoli gate");
          builder.create<quake::XOp>(loc, false, empty, controls, targets);
        }},
       {"fredkin",
        [&]() {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 2 && "ill-formed fredkin gate");
          builder.create<quake::SwapOp>(loc, false, empty, controls, targets);
        }},
       {"rccx",
        // Rccx q1, q2, q3:
        //  Cz q1, q3
        //  H q3
        //  T q3
        //  Cx q2, q3
        //  Tdg q3
        //  Cx q1, q3
        //  T 3
        //  Cx q2, q3
        //  Tdg q3
        //  H q3
        [&]() {
          assert(params.empty() && controls.size() == 2 &&
                 targets.size() == 1 && "ill-formed rccx gate");
          auto q1 = controls[0];
          auto q2 = controls[1];
          auto q3 = targets[0];

          builder.create<quake::ZOp>(loc, false, empty, q1, q3);
          builder.create<quake::HOp>(loc, false, empty, empty, q3);
          builder.create<quake::TOp>(loc, false, empty, empty, q3);
          builder.create<quake::XOp>(loc, false, empty, q2, q3);
          builder.create<quake::TOp>(loc, true, empty, empty, q3);
          builder.create<quake::XOp>(loc, false, empty, q1, q3);
          builder.create<quake::TOp>(loc, false, empty, empty, q3);
          builder.create<quake::XOp>(loc, false, empty, q2, q3);
          builder.create<quake::TOp>(loc, true, empty, empty, q3);
          builder.create<quake::HOp>(loc, false, empty, empty, q3);
        }},
       {"rxx",
        // Rxx(θ) q1, q2:
        //  H q1
        //  H q2
        //  Cx q1, q2
        //  Rz(θ) q2
        //  Cx q1, q2
        //  H q2
        //  H q1
        [&]() {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 2 && "ill-formed rxx gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto theta = params[0];
          builder.create<quake::HOp>(loc, false, empty, empty, targets);
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
          builder.create<quake::RzOp>(loc, false, theta, empty, q2);
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
          builder.create<quake::HOp>(loc, false, empty, empty, targets);
        }},
       {"ryy",
        // Ryy(θ) q1, q2:
        //  Rx(π/2) q1
        //  Rx(π/2) q2
        //  Cx q1, q2
        //  Rz(θ) q2
        //  Cx q1, q2
        //  Rx(-π/2) q2
        //  Rx(-π/2) q1
        [&]() {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 2 && "ill-formed ryy gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto param1 = plusHalfPi;
          auto param2 = params[0];
          auto param3 = minusHalfPi;

          builder.create<quake::RxOp>(loc, false, param1, empty, targets);
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
          builder.create<quake::RzOp>(loc, false, param2, empty, q2);
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
          builder.create<quake::RxOp>(loc, false, param3, empty, targets);
        }},
       {"rzz",
        // Rzz(θ) q1, q2:
        //  Cx q1, q2
        //  Rz(θ) q2
        //  Cx q1, q2
        [&]() {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 2 && "ill-formed rzz gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto theta = params[0];
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
          builder.create<quake::RzOp>(loc, false, theta, empty, q2);
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
        }},
       {"rzx",
        // Rzz(θ) q1, q2:
        //  H q1
        //  Cx q1, q2
        //  Rx(θ) q2
        //  Cx q1, q2
        //  H q1
        [&]() {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 2 && "ill-formed rzx gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto theta = params[0];
          builder.create<quake::HOp>(loc, false, empty, empty, q1);
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
          builder.create<quake::RxOp>(loc, false, theta, empty, q2);
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
          builder.create<quake::HOp>(loc, false, empty, empty, q1);
        }},
       {"dcx",
        // Dcx q1, q2:
        //  Cx q1, q2
        //  Cx q2, q1
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 2 &&
                 !adj && "ill-formed dcx gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
          builder.create<quake::XOp>(loc, false, empty, q2, q1);
        }},
       {"ecr",
        // Ecr q1, q2:
        //  S q1
        //  Rx(π/2) q2
        //  Cx q1, q2
        //  X q1
        [&]() {
          assert(params.empty() && controls.empty() && targets.size() == 2 &&
                 !adj && "ill-formed ecr gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto param1 = plusHalfPi;
          builder.create<quake::SOp>(loc, false, empty, empty, q1);
          builder.create<quake::RxOp>(loc, false, param1, empty, q2);
          builder.create<quake::XOp>(loc, false, empty, q1, q2);
          builder.create<quake::XOp>(loc, false, empty, empty, q1);
        }}};
  auto it = gateMap.find(gateId);
  if (it != gateMap.end()) {
    it->second();
  } else {
    assert(false && ("Unknown gate: " + gateId + "\n").c_str());
  }
}