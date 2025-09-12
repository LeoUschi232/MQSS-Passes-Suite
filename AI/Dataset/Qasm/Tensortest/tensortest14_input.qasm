OPENQASM 2.0;
include "qelib1.inc";
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
qreg q[5];
creg meas[5];
cry(1.1535088273471947) q[0],q[4];
csdg q[4],q[1];
rzz(0.9767564727879403) q[2],q[0];
z q[0];
rx(0.15178212168394944) q[0];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
