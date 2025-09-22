OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
qreg q[5];
creg meas[5];
ry(-1.8873290565959517) q[2];
cy q[1],q[3];
r(2.530651909764689,-2.094114403851137) q[0];
cs q[2],q[4];
csdg q[1],q[0];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
