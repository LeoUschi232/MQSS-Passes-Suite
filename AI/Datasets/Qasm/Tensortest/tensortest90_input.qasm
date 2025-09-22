OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
qreg q[5];
creg meas[5];
tdg q[2];
cy q[4],q[2];
ryy(0.47186387318513656) q[1],q[2];
ch q[3],q[1];
crz(-2.758635566405428) q[0],q[2];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
