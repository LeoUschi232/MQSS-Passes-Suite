OPENQASM 2.0;
include "qelib1.inc";
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
qreg q[5];
creg meas[5];
iswap q[2],q[3];
ccz q[4],q[3],q[0];
rzx(2.945090278800097) q[4],q[1];
y q[3];
iswap q[4],q[3];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
