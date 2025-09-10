OPENQASM 2.0;
include "qelib1.inc";
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
qreg q[5];
creg meas[5];
cs q[0],q[4];
xx_minus_yy(1.8674737339480743,0) q[4],q[0];
sdg q[3];
cry(-0.8799120884857845) q[0],q[2];
s q[1];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
