OPENQASM 2.0;
include "qelib1.inc";
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
qreg q[5];
creg meas[5];
rzz(-0.39216711169055163) q[3],q[1];
x q[4];
ecr q[0],q[3];
r(1.614315898206777,-0.267994644596663) q[2];
r(-2.955440692190405,2.3067338999485756) q[4];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
