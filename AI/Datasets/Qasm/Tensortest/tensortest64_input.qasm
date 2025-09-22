OPENQASM 2.0;
include "qelib1.inc";
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
qreg q[5];
creg meas[5];
ry(-0.04901254673389843) q[0];
rz(-3.0944349224691066) q[4];
ecr q[3],q[4];
iswap q[1],q[2];
ch q[4],q[2];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
