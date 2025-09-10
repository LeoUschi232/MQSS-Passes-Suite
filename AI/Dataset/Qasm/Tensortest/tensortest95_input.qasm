OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
qreg q[5];
creg meas[5];
rzz(-2.1277432312810935) q[2],q[4];
swap q[0],q[1];
rzx(0.8929606107478607) q[1],q[2];
xx_plus_yy(-2.5728801858997556,0) q[3],q[2];
ecr q[4],q[1];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
