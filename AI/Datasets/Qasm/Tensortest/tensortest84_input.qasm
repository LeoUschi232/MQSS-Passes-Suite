OPENQASM 2.0;
include "qelib1.inc";
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
qreg q[5];
creg meas[5];
iswap q[2],q[3];
cs q[3],q[4];
ccx q[3],q[4],q[2];
tdg q[1];
ch q[3],q[1];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
