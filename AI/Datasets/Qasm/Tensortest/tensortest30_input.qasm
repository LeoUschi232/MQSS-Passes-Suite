OPENQASM 2.0;
include "qelib1.inc";
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
qreg q[5];
creg meas[5];
crz(0.01829244884392045) q[1],q[3];
s q[4];
csdg q[2],q[0];
r(-2.539624349044395,-1.977929698673157) q[1];
tdg q[2];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
