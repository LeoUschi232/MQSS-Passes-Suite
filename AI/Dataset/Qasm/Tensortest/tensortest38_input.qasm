OPENQASM 2.0;
include "qelib1.inc";
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
qreg q[5];
creg meas[5];
cy q[0],q[3];
ccz q[4],q[0],q[3];
ry(1.158125510785923) q[0];
rzz(1.333287855528983) q[3],q[0];
swap q[2],q[0];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
