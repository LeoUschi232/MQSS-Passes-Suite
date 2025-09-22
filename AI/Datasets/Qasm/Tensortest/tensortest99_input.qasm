OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
qreg q[5];
creg meas[5];
r(-2.0186567447599586,-0.9532674485400272) q[4];
cry(0.06561350898102392) q[0],q[1];
h q[0];
cx q[0],q[2];
ccx q[2],q[3],q[0];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
