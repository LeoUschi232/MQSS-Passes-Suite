OPENQASM 2.0;
include "qelib1.inc";
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
qreg q[5];
creg meas[5];
tdg q[2];
cx q[4],q[3];
ecr q[2],q[0];
ccx q[1],q[2],q[0];
t q[3];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
