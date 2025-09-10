OPENQASM 2.0;
include "qelib1.inc";
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
qreg q[5];
creg meas[5];
iswap q[4],q[2];
crz(-0.9420606384502999) q[2],q[1];
cry(-0.4024876272509128) q[4],q[2];
rzx(-0.3173408080658344) q[3],q[4];
rz(-1.9534766381998763) q[3];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
