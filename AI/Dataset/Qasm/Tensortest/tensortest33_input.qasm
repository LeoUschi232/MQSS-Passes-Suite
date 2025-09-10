OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
qreg q[5];
creg meas[5];
rz(-1.3620861206884802) q[1];
r(-1.0441645397388077,-1.117277940103703) q[1];
ecr q[0],q[3];
tdg q[2];
rzx(2.188578371502026) q[2],q[3];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
