OPENQASM 2.0;
include "qelib1.inc";
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
qreg q[5];
creg meas[5];
xx_minus_yy(-2.0455861657150978,0) q[3],q[2];
ry(0.9535615213088233) q[2];
csdg q[1],q[2];
r(-0.4254350923673553,-1.220883148702497) q[0];
y q[0];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
