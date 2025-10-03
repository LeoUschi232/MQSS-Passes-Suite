OPENQASM 2.0;
include "qelib1.inc";
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
qreg q[5];
creg meas[5];
ccz q[3],q[2],q[4];
cz q[1],q[3];
xx_minus_yy(-1.006344951105636,0) q[0],q[2];
rxx(0.21586422635314761) q[2],q[0];
ecr q[2],q[0];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
