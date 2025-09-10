OPENQASM 2.0;
include "qelib1.inc";
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
qreg q[5];
creg meas[5];
cx q[2],q[1];
rz(2.2829286731343954) q[2];
ecr q[3],q[0];
r(-2.248960398155755,0.9869857168119847) q[4];
rzx(1.3373296868436313) q[4],q[1];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
