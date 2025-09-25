OPENQASM 2.0;
include "qelib1.inc";
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
qreg q[5];
creg meas[5];
xx_plus_yy(-2.598341665972772,0) q[0],q[1];
cry(-2.525225506934188) q[4],q[1];
crz(-0.7989985443469636) q[2],q[4];
xx_plus_yy(-0.0866857116882982,0) q[3],q[4];
cz q[0],q[2];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
