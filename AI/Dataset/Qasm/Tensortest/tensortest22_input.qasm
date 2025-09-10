OPENQASM 2.0;
include "qelib1.inc";
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
qreg q[5];
creg meas[5];
cs q[3],q[1];
xx_plus_yy(2.778568451254385,0) q[2],q[0];
ccx q[4],q[0],q[2];
ryy(0.3823571771636942) q[4],q[2];
ryy(2.6138194018540135) q[4],q[0];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
