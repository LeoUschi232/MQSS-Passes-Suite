OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
qreg q[5];
creg meas[5];
cy q[1],q[4];
r(0.9047139164031988,-1.622921759941552) q[4];
cs q[2],q[4];
swap q[3],q[4];
ccz q[1],q[4],q[3];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
