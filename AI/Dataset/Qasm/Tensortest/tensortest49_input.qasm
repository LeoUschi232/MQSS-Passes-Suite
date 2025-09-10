OPENQASM 2.0;
include "qelib1.inc";
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
qreg q[5];
creg meas[5];
ecr q[1],q[2];
cry(3.0358963002923725) q[4],q[3];
r(-0.7736375586416742,2.264758267643262) q[4];
r(-1.0973341914278572,-1.5079802719666666) q[4];
h q[2];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
